"""Instruction-format dataset contract, produced in Stage 3 from `VulnSample`."""

from pydantic import BaseModel


class InstructionExample(BaseModel):
    id: str
    sample_id: str  # references VulnSample.id
    prompt: str  # fully formatted instruction
    target_cwe: str
    target_severity: str
    target_explanation: str
    target_patch_diff: str | None = None
    token_count_estimate: int
