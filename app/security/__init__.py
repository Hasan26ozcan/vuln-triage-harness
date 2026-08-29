"""Security utilities for the vulnerability triage harness.

Provides path validation to prevent filesystem-escape attacks when CLI
arguments are used to construct file paths.
"""

from app.security.paths import (
    PathSecurityError,
    get_project_root,
    is_hf_model_id,
    safe_read_text,
    validate_output_path,
    validate_path,
    validate_path_str,
)

__all__ = [
    "PathSecurityError",
    "get_project_root",
    "is_hf_model_id",
    "safe_read_text",
    "validate_output_path",
    "validate_path",
    "validate_path_str",
]
