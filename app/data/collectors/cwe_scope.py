"""CWE scope for this project — the single source of truth for Stage 1.

The roadmap is explicit about this: stay narrow (5-8 CWE classes). A wider
scope dilutes an already-small dataset and makes every downstream number
(macro-F1, exec-pass-rate) noisier and harder to defend under questioning.

Each entry also carries the target language, because the exec-based eval
sandbox (Stage 6, Tier 3) is built per-language — mixing languages here
means mixing sandbox setups later.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CweSpec:
    cwe_id: str
    name: str
    language: str
    min_samples: int  # Stage 1 DoD: 80-150 per class


# Deliberately capped at 6 classes. All target Python or JavaScript so
# Stage 6 Tier 3 only needs one language's sandbox image to start
# (roadmap risk mitigation: "limit to one language first, expand later").
CWE_SCOPE: tuple[CweSpec, ...] = (
    CweSpec("CWE-89", "SQL Injection", "python", min_samples=80),
    CweSpec("CWE-79", "Cross-Site Scripting", "javascript", min_samples=80),
    CweSpec("CWE-22", "Path Traversal", "python", min_samples=80),
    CweSpec("CWE-78", "OS Command Injection", "python", min_samples=80),
    CweSpec("CWE-190", "Integer Overflow", "python", min_samples=80),
    CweSpec("CWE-502", "Deserialization of Untrusted Data", "python", min_samples=80),
)

CWE_IDS: frozenset[str] = frozenset(spec.cwe_id for spec in CWE_SCOPE)


def cwe_spec(cwe_id: str) -> CweSpec | None:
    """Look up the spec for a CWE ID, or None if it's out of scope."""
    for spec in CWE_SCOPE:
        if spec.cwe_id == cwe_id:
            return spec
    return None


def in_scope(cwe_id: str) -> bool:
    return cwe_id in CWE_IDS
