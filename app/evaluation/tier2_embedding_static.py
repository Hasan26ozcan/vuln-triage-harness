"""Stage 6, Tier 2 — static-analysis signal + embedding similarity.

This tier combines two signals:

1. **Semgrep rule mapping** — each finding in ``VulnSample.static_findings``
   carries a ``rule_id`` (collected in Stage 1 via the bundled Semgrep rule
   pack). We map that rule ID to a CWE class and vote across all findings.
   This requires **no** model download — it's a static lookup table.

2. **Patch embedding similarity** *(optional)* — when a sentence-embedding
   model is available, we compute the cosine similarity between the model's
   ``suggested_patch_diff`` and the gold ``fixed_code``. Higher similarity
   → higher confidence that the patch is correct. The embedding import is
   lazy: if ``sentence-transformers`` is not installed, the evaluator runs
   in static-only mode and ``embedding_similarity`` is ``None``.

Tier 2's output (``Tier2Result``) can be compared against both the gold CWE
(accuracy of the static signal) and the model's prediction (corroboration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.schemas.prediction_eval import ModelPrediction, Tier2Result
from app.schemas.vuln import VulnSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule ID → CWE mapping
#
# Derived from the ``metadata.cwe`` field of the bundled Semgrep rules in
# ``app/data/collectors/rules/{python,javascript}.yaml`` plus the rule IDs
# used in the gold-eval set. In both cases the rule ID encodes the vulnerability
# type, and the CWE is recorded in the rule metadata.
# ---------------------------------------------------------------------------

DEFAULT_RULE_TO_CWE: dict[str, str] = {
    # --- CWE-89: SQL Injection ---
    "python.sqli-string-concat": "CWE-89",
    "python.sqli-f-string": "CWE-89",
    "python.sqli-taint-assign-then-execute": "CWE-89",
    "javascript.sqli-string-concat": "CWE-89",
    # --- CWE-79: XSS ---
    "javascript.dom-xss-innerhtml": "CWE-79",
    "javascript.xss.document-write-concat": "CWE-79",
    "javascript.xss-innerhtml": "CWE-79",
    # --- CWE-22: Path Traversal ---
    "python.path-traversal": "CWE-22",
    "python.path-traversal-open": "CWE-22",
    "python.path-traversal-unsafe-open": "CWE-22",
    # --- CWE-78: OS Command Injection ---
    "python.command-injection": "CWE-78",
    "python.command-injection-popen": "CWE-78",
    "python.os-command-injection": "CWE-78",
    # --- CWE-190: Integer Overflow ---
    "python.integer-overflow": "CWE-190",
    "python.integer-overflow-unchecked": "CWE-190",
    "python.integer-overflow-bitshift": "CWE-190",
    # --- CWE-502: Deserialization ---
    "python.deserialization-pickle": "CWE-502",
    "python.deserialization-yaml": "CWE-502",
    "python.unsafe-deserialization": "CWE-502",
}


@dataclass
class EmbeddingBackend:
    """Thin wrapper around a sentence-transformers embedding model.

    Imported lazily so that Tier 2 works in static-only mode without
    ``sentence-transformers`` installed. Tests inject a simple object
    implementing ``encode(text) -> list[float]``.
    """

    model_name: str = "intfloat/multilingual-e5-base"
    _model = None

    def encode(self, text: str) -> list[float]:
        """Return the embedding for *text*.

        The model is loaded on first use. If ``sentence-transformers``
        is not installed, a ``RuntimeError`` is raised.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # noqa: SIM110
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Run `pip install -e '.[ml]'` for embedding-based eval, "
                    "or run Tier 2 in static-only mode."
                ) from exc
            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(text).tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class StaticSignalEvaluator:
    """Tier 2 evaluator — Semgrep finding mapping + optional embedding similarity.

    Parameters
    ----------
    rule_to_cwe:
        Custom mapping from Semgrep ``rule_id`` to CWE ID. Defaults to
        ``DEFAULT_RULE_TO_CWE``. Injecting this lets tests verify mapping
        logic without relying on the default table.
    embedding_model:
        Optional sentence-transformers model name. When set, the evaluator
        computes cosine similarity between the predicted patch and the gold
        fix. When ``None`` (the default), embedding similarity is skipped
        and ``Tier2Result.embedding_similarity`` is ``None``.
    """

    def __init__(
        self,
        rule_to_cwe: dict[str, str] | None = None,
        embedding_model: str | None = None,
    ):
        self._rule_to_cwe = dict(rule_to_cwe) if rule_to_cwe else dict(DEFAULT_RULE_TO_CWE)
        self._embedder: EmbeddingBackend | None = (
            EmbeddingBackend(model_name=embedding_model) if embedding_model else None
        )

    @property
    def rule_to_cwe(self) -> dict[str, str]:
        return dict(self._rule_to_cwe)

    @property
    def uses_embeddings(self) -> bool:
        return self._embedder is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        sample: VulnSample,
        prediction: ModelPrediction | None = None,
    ) -> Tier2Result:
        """Evaluate a sample using static-analysis signal.

        Parameters
        ----------
        sample:
            The ``VulnSample`` with ``static_findings`` populated (from Stage 1).
        prediction:
            Optional ``ModelPrediction`` — when provided and an embedding model
            is configured, cosine similarity between the predicted patch and
            the gold ``fixed_code`` is computed.
        """
        # 1. Vote across Semgrep findings.
        cwe_votes: dict[str, int] = {}
        signal_sources: list[str] = []

        for finding in sample.static_findings:
            cwe = self._rule_to_cwe.get(finding.rule_id)
            signal_sources.append(f"semgrep:{finding.rule_id}")
            if cwe is not None:
                cwe_votes[cwe] = cwe_votes.get(cwe, 0) + 1

        if cwe_votes:
            predicted_cwe = max(cwe_votes, key=lambda k: cwe_votes[k])
            total_votes = sum(cwe_votes.values())
            confidence = cwe_votes[predicted_cwe] / total_votes if total_votes else 0.0
        else:
            predicted_cwe = None
            confidence = 0.0

        # 2. Optional embedding similarity (between predicted patch and gold fix).
        embedding_similarity = self._compute_embedding_similarity(sample, prediction)

        return Tier2Result(
            sample_id=sample.id,
            predicted_cwe=predicted_cwe,
            confidence=round(confidence, 4),
            signal_sources=signal_sources,
            embedding_similarity=embedding_similarity,
        )

    def evaluate_all(
        self,
        samples: list[VulnSample],
        predictions: dict[str, ModelPrediction] | None = None,
    ) -> list[Tier2Result]:
        """Evaluate a batch of samples.

        ``predictions`` maps ``sample_id`` → ``ModelPrediction`` so the
        embedding similarity can be computed per sample.
        """
        pred_map = predictions or {}
        return [self.evaluate(s, pred_map.get(s.id)) for s in samples]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_embedding_similarity(
        self,
        sample: VulnSample,
        prediction: ModelPrediction | None,
    ) -> float | None:
        """Cosine similarity between the model's patch and the gold fix.

        Returns ``None`` when no embedding model is configured, the sample
        has no ``fixed_code``, or the prediction has no patch.
        """
        if self._embedder is None:
            return None
        if sample.fixed_code is None:
            return None
        if prediction is None or not prediction.suggested_patch_diff.strip():
            return None

        try:
            emb_pred = self._embedder.encode(prediction.suggested_patch_diff)
            emb_gold = self._embedder.encode(sample.fixed_code)
            sim = _cosine_similarity(emb_pred, emb_gold)
            return round(sim, 4)
        except RuntimeError as exc:
            logger.warning("Embedding similarity skipped: %s", exc)
            return None
