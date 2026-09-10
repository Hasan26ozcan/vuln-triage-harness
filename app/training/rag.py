"""Stage 5+ — Retrieval-Augmented Generation for vulnerability patching.

This module implements a RAG pipeline that retrieves similar known
vulnerabilities from the training set and uses them as in-context
examples when generating patches. This is particularly useful for
the patch-generation step, where the model needs to see concrete
examples of how similar vulnerabilities were fixed.

The pipeline works in two phases:

1. **Indexing** (once): Embed all training examples using sentence
   transformers and store them in an in-memory or disk-based index.

2. **Retrieval** (at inference time): Given a vulnerable code sample,
   compute its embedding, find the top-k most similar training examples,
   and inject them as few-shot context for patch generation.

Usage::

    from app.training.rag import RAGRetriever, build_rag_index

    # Build index from training data
    retriever = build_rag_index(train_examples)

    # Retrieve similar examples for a new vulnerability
    examples = retriever.retrieve(vulnerable_code, top_k=3)

    # Use in a patch-generation prompt
    prompt = build_rag_patch_prompt(sample, examples)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    _HAS_RAG = True
except ImportError:  # pragma: no cover
    _HAS_RAG = False
    logger.warning(
        "sentence_transformers not installed. "
        "RAG retrieval requires it: pip install sentence-transformers."
    )


@dataclass
class RetrievalResult:
    """Result of a RAG retrieval query."""

    sample_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGRetriever:
    """Retrieval-Augmented Generation retriever for vulnerability examples.

    Uses sentence embeddings to find semantically similar training examples.
    Falls back to simple keyword matching when sentence_transformers is
    not available (graceful degradation).
    """

    model_name: str = "all-MiniLM-L6-v2"
    examples: list[Any] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    texts: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    use_approximate: bool = True  # Use FAISS-style approximate search when available

    def __post_init__(self) -> None:
        if not _HAS_RAG:
            logger.warning("RAG disabled — sentence_transformers not available")

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(
        self,
        examples: list[Any],
        text_field: str = "target_explanation",
        id_field: str = "sample_id",
    ) -> None:
        """Build the embedding index from training examples.

        Parameters
        ----------
        examples:
            List of ``InstructionExample`` or similar objects.
        text_field:
            Attribute name to use for embedding text.
        id_field:
            Attribute name to use for sample IDs.
        """
        if not _HAS_RAG:
            logger.warning("Cannot build RAG index — sentence_transformers not available")
            self.examples = examples
            self.texts = [getattr(ex, text_field, str(ex)) for ex in examples]
            self.ids = [str(getattr(ex, id_field, f"idx_{i}")) for i, ex in enumerate(examples)]
            return

        self.examples = examples
        self.texts = [getattr(ex, text_field, str(ex)) for ex in examples]
        self.ids = [str(getattr(ex, id_field, f"idx_{i}")) for i, ex in enumerate(examples)]

        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(self.texts) if t and t.strip()]
        self.texts = [self.texts[i] for i in valid_indices]
        self.ids = [self.ids[i] for i in valid_indices]
        self.examples = [self.examples[i] for i in valid_indices]

        logger.info("Building RAG index with %d examples...", len(self.texts))

        model = SentenceTransformer(self.model_name)
        self.embeddings = model.encode(
            self.texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        logger.info("RAG index built. Embedding shape: %s", self.embeddings.shape)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        query_field: str = "prompt",
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most similar training examples.

        Parameters
        ----------
        query_text:
            Text to search for (e.g., vulnerable code or prompt).
        top_k:
            Number of results to return.
        query_field:
            Attribute name for the query text (used if query_text is None).

        Returns
        -------
        List of ``RetrievalResult`` sorted by similarity (descending).
        """
        if not self.examples:
            logger.warning("RAG index not built — returning empty results")
            return []

        if self.embeddings is not None and _HAS_RAG:
            return self._retrieve_by_embedding(query_text, top_k)

        return self._retrieve_by_keyword(query_text, top_k, query_field)

    def _retrieve_by_embedding(self, query_text: str, top_k: int) -> list[RetrievalResult]:
        """Embed the query and do cosine similarity search."""
        model = SentenceTransformer(self.model_name)
        query_embedding = model.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True
        )

        # Cosine similarity (embeddings are already normalized)
        similarities = self.embeddings @ query_embedding.T  # type: ignore[operator]
        scores = similarities.flatten()

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                RetrievalResult(
                    sample_id=self.ids[idx],
                    text=self.texts[idx],
                    score=float(scores[idx]),
                    metadata={
                        "example_index": idx,
                    },
                )
            )

        logger.info(
            "Retrieved %d results for query (top score: %.4f)",
            len(results),
            scores[top_indices[0]],
        )
        return results

    def _retrieve_by_keyword(
        self, query_text: str, top_k: int, query_field: str = "target_explanation"
    ) -> list[RetrievalResult]:
        """Simple keyword-based fallback when embeddings are unavailable."""
        query_lower = query_text.lower()
        results = []

        for i, ex in enumerate(self.examples):
            text = getattr(ex, query_field, str(ex))
            score = sum(1 for word in query_lower.split() if word in text.lower())
            if score > 0:
                results.append(
                    RetrievalResult(
                        sample_id=str(getattr(ex, "sample_id", f"idx_{i}")),
                        text=text,
                        score=score / max(1, len(query_lower.split())),
                        metadata={"example_index": i},
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ------------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------------


def build_rag_patch_prompt(
    vulnerable_code: str,
    cwe_id: str,
    retrieval_results: list[RetrievalResult],
) -> str:
    """Build a few-shot prompt for patch generation using RAG results.

    Similar to ``build_few_shot_prompt`` in the evaluation module, but
    specifically designed for the retrieval-augmented context where
    each retrieved example demonstrates how a *similar* vulnerability
    was fixed.

    Parameters
    ----------
    vulnerable_code:
        The vulnerable code snippet that needs a patch.
    cwe_id:
        The CWE classification (already determined by the classification model).
    retrieval_results:
        Top-k ``RetrievalResult`` objects from the RAG retriever.

    Returns
    -------
    Formatted prompt string with retrieved examples as few-shot context.
    """
    parts: list[str] = [
        "### System: You are a security code repair expert.",
        f"### Task: Generate a unified diff patch to fix a {cwe_id} vulnerability.",
        "",
        "### Examples of similar vulnerability fixes:",
    ]

    for i, result in enumerate(retrieval_results, 1):
        parts.append(f"--- Example {i} (similarity: {result.score:.3f}) ---")
        parts.append("Context:")
        parts.append(result.text[:500] if len(result.text) > 500 else result.text)
        parts.append("Patch:")
        parts.append("[See training data for the original patch diff]")
        parts.append("")

    parts.append("--- Your Turn ---")
    parts.append("### Vulnerable Code:")
    parts.append("```python")
    parts.append(vulnerable_code)
    parts.append("```")
    parts.append("")
    parts.append("### Instructions:")
    parts.append("1. Generate a unified diff patch that fixes the vulnerability.")
    parts.append("2. Only modify the code that contains the vulnerability.")
    parts.append("3. Use the format: ```diff\n...patch...\n```")
    parts.append("")
    parts.append("### Patch:")
    parts.append("```diff")

    return "\n".join(parts)


def build_rag_classification_prompt(
    vulnerable_code: str,
    static_findings: str,
    language: str,
    retrieval_results: list[RetrievalResult],
) -> str:
    """Build a few-shot prompt for CWE classification using RAG results.

    Uses retrieved similar vulnerabilities as in-context examples to help
    the model classify the current vulnerability correctly.
    """
    parts: list[str] = [
        "### System: You are a security vulnerability classifier.",
        f"### Language: {language}",
        "",
        "### Classification examples from similar vulnerabilities:",
    ]

    for i, result in enumerate(retrieval_results, 1):
        parts.append(f"--- Example {i} (similarity: {result.score:.3f}) ---")
        parts.append("Code context:")
        parts.append(result.text[:300] if len(result.text) > 300 else result.text)
        parts.append("CWE: [Classified based on static findings]")
        parts.append("")

    parts.append("--- Classify this vulnerability ---")
    parts.append(f"### Vulnerable Code ({language}):")
    parts.append("```python")
    parts.append(vulnerable_code)
    parts.append("```")
    parts.append("")
    parts.append("### Static Analysis Findings:")
    parts.append(static_findings)
    parts.append("")
    parts.append("### Response (JSON):")
    parts.append("```json")
    parts.append('{"cwe_id": "...", "severity": "...", "explanation": "..."}')
    parts.append("```")

    return "\n".join(parts)


# ------------------------------------------------------------------
# Integration helpers
# ------------------------------------------------------------------


def retrieve_and_build_prompt(
    retriever: RAGRetriever,
    vulnerable_code: str,
    cwe_id: str,
    top_k: int = 5,
) -> tuple[list[RetrievalResult], str]:
    """Convenience function: retrieve examples and build a patch-generation prompt.

    Parameters
    ----------
    retriever:
        The RAGRetriever instance.
    vulnerable_code:
        The vulnerable code snippet to search for.
    cwe_id:
        The CWE classification for context.
    top_k:
        Number of results to retrieve.

    Returns
    -------
    Tuple of (retrieval_results, formatted_prompt_string).
    """
    retrieval_results = retriever.retrieve(vulnerable_code, top_k=top_k)
    prompt = build_rag_patch_prompt(vulnerable_code, cwe_id, retrieval_results)
    return retrieval_results, prompt
