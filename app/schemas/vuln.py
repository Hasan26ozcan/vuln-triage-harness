"""Raw vulnerability sample contracts.

These are the schemas produced by Stage 1 (data collection) and consumed
by Stage 2 (cleaning / leakage-safe split). `repo_name` and `commit_sha`
are mandatory because the leakage-safe split in Stage 2 groups examples
by repository, not by individual sample.
"""

from typing import Literal

from pydantic import BaseModel, Field


class StaticFinding(BaseModel):
    """A single Semgrep (or other static analyzer) finding on a code sample."""

    tool: Literal["semgrep"]
    rule_id: str
    message: str
    line_range: tuple[int, int]


class VulnSample(BaseModel):
    """A single vulnerable/fixed code pair enriched with CWE, severity, and
    static-analysis signal.
    """

    id: str
    source: Literal["cve_real", "synthetic_injected", "ctf_sample"]
    repo_name: str = Field(..., description="Required for leakage-safe split (Stage 2).")
    commit_sha: str | None = None
    cve_id: str | None = None
    cwe_id: str = Field(..., description='e.g. "CWE-89"')
    severity: Literal["low", "medium", "high", "critical"]
    language: str
    vulnerable_code: str
    fixed_code: str | None = None
    static_findings: list[StaticFinding] = []
    description: str
    split: Literal["train", "val", "test", "gold_eval"] | None = Field(
        default=None, description="Assigned in Stage 2, immutable afterwards."
    )
