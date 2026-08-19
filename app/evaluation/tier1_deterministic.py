"""Stage 6, Tier 1 — deterministic CWE classification.

This tier is a pure-Python, zero-dependency rule engine: it matches regex
patterns against the vulnerable code to predict the CWE class. No model, no
Semgrep, no Docker — just string matching.

It serves three purposes:
  1. **Floor baseline** — what can a simple heuristic achieve? Everything above
     this is the model's marginal contribution.
  2. **Fast pre-filter** — Tier 2/3/4 only run on samples Tier 1 couldn't
     resolve (configurable in the runner).
  3. **Sanity check** — if the model predicts CWE-89 for a sample Tier 1 also
     flags as CWE-79, that's worth investigating.

The rules are injectable — tests pass a custom rule list to verify scoring
without relying on the default patterns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.schemas.prediction_eval import Tier1Result
from app.schemas.vuln import VulnSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatternRule:
    """A single regex rule that fires for a CWE class.

    Attributes
    ----------
    cwe:
        The CWE ID this rule predicts (e.g. ``"CWE-89"``).
    pattern:
        A regex string compiled with ``re.IGNORECASE | re.DOTALL``.
    confidence:
        How confident the rule is when it fires (0.0–1.0). Higher = more
        specific / less likely to false-positive.
    description:
        Human-readable label stored in ``Tier1Result.matched_pattern``.
    """

    cwe: str
    pattern: str
    confidence: float
    description: str


# ---------------------------------------------------------------------------
# Default rule set — derived from the gold-eval samples and the Semgrep rules
# in app/data/collectors/rules/{python,javascript}.yaml.
#
# Each rule maps a code-level pattern to a CWE class. Confidence reflects
# how specific the pattern is: a precise API call (``pickle.loads``) gets
# 0.99; a broad syntactic marker (bit-shift) gets 0.6.
# ---------------------------------------------------------------------------

DEFAULT_TIER1_RULES: tuple[PatternRule, ...] = (
    # --- CWE-89: SQL Injection ---
    PatternRule(
        cwe="CWE-89",
        pattern=r"\.execute\s*\(\s*f[\"']",
        confidence=0.9,
        description="f-string passed to cursor.execute()",
    ),
    PatternRule(
        cwe="CWE-89",
        pattern=r"\.execute\s*\(\s*[\"'][^\"']*[\"']\s*\+",
        confidence=0.85,
        description="string concatenation inside execute() call",
    ),
    PatternRule(
        cwe="CWE-89",
        pattern=r"\.execute\s*\(\s*[\"'].*?%.*[\"']\s*%",
        confidence=0.8,
        description="printf-style formatting inside execute() call",
    ),
    PatternRule(
        cwe="CWE-89",
        pattern=r"SELECT\s+.*\+.*",
        confidence=0.75,
        description="string concatenation in SQL SELECT query",
    ),
    PatternRule(
        cwe="CWE-89",
        pattern=r"f[\"']SELECT",
        confidence=0.85,
        description="f-string interpolation in SQL SELECT query",
    ),
    # --- CWE-79: XSS ---
    PatternRule(
        cwe="CWE-79",
        pattern=r"\.innerHTML\s*=\s*",
        confidence=0.95,
        description="assignment to innerHTML (XSS sink)",
    ),
    PatternRule(
        cwe="CWE-79",
        pattern=r"\.outerHTML\s*=\s*",
        confidence=0.9,
        description="assignment to outerHTML (XSS sink)",
    ),
    PatternRule(
        cwe="CWE-79",
        pattern=r"document\.write\s*\(",
        confidence=0.85,
        description="document.write() with untrusted input",
    ),
    PatternRule(
        cwe="CWE-79",
        pattern=r"\.send\s*\(\s*[\"'].*?\+.*?[\"']",
        confidence=0.7,
        description="HTML response built via string concatenation",
    ),
    # --- CWE-22: Path Traversal ---
    PatternRule(
        cwe="CWE-22",
        pattern=r"open\s*\([^)]*\+[^)]*\)",
        confidence=0.85,
        description="path concatenation inside open()",
    ),
    PatternRule(
        cwe="CWE-22",
        pattern=r"open\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*,",
        confidence=0.7,
        description="non-literal path passed to open() (possible traversal)",
    ),
    # --- CWE-78: OS Command Injection ---
    PatternRule(
        cwe="CWE-78",
        pattern=r"shell\s*=\s*True",
        confidence=0.95,
        description="shell=True in subprocess call",
    ),
    PatternRule(
        cwe="CWE-78",
        pattern=r"os\.system\s*\(",
        confidence=0.9,
        description="os.system() with potentially user-controlled input",
    ),
    PatternRule(
        cwe="CWE-78",
        pattern=r"os\.popen\s*\(",
        confidence=0.85,
        description="os.popen() with potentially user-controlled input",
    ),
    PatternRule(
        cwe="CWE-78",
        pattern=r"subprocess\.run\s*\(\s*[\"'].*?\+",
        confidence=0.8,
        description="string concatenation in subprocess.run()",
    ),
    # --- CWE-190: Integer Overflow / Wraparound ---
    PatternRule(
        cwe="CWE-190",
        pattern=r"bytearray\s*\(",
        confidence=0.7,
        description="bytearray allocation from untrusted size",
    ),
    PatternRule(
        cwe="CWE-190",
        pattern=r"\[0\s*\]\s*\*\s*",
        confidence=0.65,
        description="list multiplication with potentially large size",
    ),
    PatternRule(
        cwe="CWE-190",
        pattern=r"<<\s*",
        confidence=0.6,
        description="unchecked bit-shift operation",
    ),
    PatternRule(
        cwe="CWE-190",
        pattern=r"\b\w+\s*\*\s*\w+",
        confidence=0.55,
        description="multiplication without bounds check",
    ),
    # --- CWE-502: Deserialization of Untrusted Data ---
    PatternRule(
        cwe="CWE-502",
        pattern=r"pickle\.loads?\s*\(",
        confidence=0.99,
        description="pickle deserialization of untrusted data",
    ),
    PatternRule(
        cwe="CWE-502",
        pattern=r"yaml\.load\s*\(",
        confidence=0.99,
        description="unsafe yaml.load() (not yaml.safe_load)",
    ),
    PatternRule(
        cwe="CWE-502",
        pattern=r"yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader\b",
        confidence=0.95,
        description="yaml.load with unsafe Loader (FullLoader/UnsafeLoader)",
    ),
)


class DeterministicEvaluator:
    """Tier 1 evaluator — regex pattern matching for CWE classification.

    Parameters
    ----------
    rules:
        A custom rule list (defaults to ``DEFAULT_TIER1_RULES``). Injecting
        rules lets tests verify scoring logic without depending on the
        default patterns.
    """

    def __init__(self, rules: list[PatternRule] | tuple[PatternRule, ...] | None = None):
        self._rules = list(rules) if rules is not None else list(DEFAULT_TIER1_RULES)
        # Pre-compile all patterns once.
        self._compiled: list[tuple[PatternRule, re.Pattern]] = [
            (rule, re.compile(rule.pattern, re.IGNORECASE | re.DOTALL)) for rule in self._rules
        ]

    @property
    def rules(self) -> list[PatternRule]:
        return list(self._rules)

    def evaluate(self, sample: VulnSample) -> Tier1Result:
        """Classify a single ``VulnSample`` using deterministic regex rules.

        Returns a ``Tier1Result``. When no rule matches, ``predicted_cwe``
        is ``None`` and ``confidence`` is 0.0.
        """
        matches: list[tuple[PatternRule, re.Match]] = []
        for rule, regex in self._compiled:
            m = regex.search(sample.vulnerable_code)
            if m:
                matches.append((rule, m))

        if not matches:
            return Tier1Result(
                sample_id=sample.id,
                predicted_cwe=None,
                confidence=0.0,
                matched_pattern=None,
                num_patterns_matched=0,
            )

        # Pick the highest-confidence match. On ties, prefer the more specific
        # (longer) pattern description to avoid generic patterns shadowing.
        matches.sort(
            key=lambda x: (
                -x[0].confidence,
                -len(x[0].description),
            )
        )
        best_rule = matches[0][0]

        return Tier1Result(
            sample_id=sample.id,
            predicted_cwe=best_rule.cwe,
            confidence=best_rule.confidence,
            matched_pattern=best_rule.description,
            num_patterns_matched=len(matches),
        )

    def evaluate_all(self, samples: list[VulnSample]) -> list[Tier1Result]:
        """Evaluate a batch of samples (convenience wrapper)."""
        return [self.evaluate(s) for s in samples]


# ---------------------------------------------------------------------------
# Module-level convenience — matches the pattern used by metrics.py
# (standalone functions + a stateful class for reuse).
# ---------------------------------------------------------------------------


def classify_deterministic(
    vulnerable_code: str,
    sample_id: str = "unknown",
    rules: list[PatternRule] | None = None,
) -> Tier1Result:
    """One-shot deterministic classification of a code snippet.

    Useful for ad-hoc classification without constructing an evaluator.
    """
    evaluator = DeterministicEvaluator(rules=rules)
    # Build a minimal VulnSample to reuse the evaluate() logic.
    from app.schemas.vuln import VulnSample

    sample = VulnSample(
        id=sample_id,
        source="cve_real",
        repo_name="eval",
        cwe_id="CWE-89",  # placeholder — Tier 1 is unsupervised
        severity="medium",
        language="python",
        vulnerable_code=vulnerable_code,
        description="",
    )
    return evaluator.evaluate(sample)
