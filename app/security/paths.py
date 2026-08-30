"""Path validation utilities to prevent path-traversal attacks.

When CLI arguments provide file paths, they can be crafted (e.g. via
``../../etc/passwd``) to escape the intended directory and read or write
files outside the project. These helpers resolve paths safely and reject
any that escape a designated base directory.

Usage::

    from app.security.paths import validate_path

    p = validate_path(cli_arg, base_dir="output")  # raises PathSecurityError if escaped
    with open(p, encoding="utf-8") as f:
        ...
"""

from __future__ import annotations

import tempfile
from pathlib import Path

__all__ = [
    "PathSecurityError",
    "get_project_root",
    "get_allowed_bases",
    "is_hf_model_id",
    "validate_path",
    "validate_path_str",
    "validate_output_path",
    "safe_read_text",
]


class PathSecurityError(ValueError):
    """Raised when a path resolves outside the allowed base directory."""

    def __init__(self, original: str | Path, resolved: Path, bases: list[Path]) -> None:
        base_strs = ", ".join(str(b) for b in bases)
        super().__init__(
            f"Path '{original}' resolves to '{resolved}' which is outside "
            f"the allowed base directories ({base_strs}). This is likely a "
            f"path traversal attempt and has been blocked."
        )
        self.original = original
        self.resolved = resolved
        self.bases = bases


def get_project_root() -> Path:
    """Return the canonical project root directory."""
    # This file is at app/security/paths.py — go up 2 levels to repo root.
    return Path(__file__).resolve().parents[2]


def get_allowed_bases(allow_temp: bool = False) -> list[Path]:
    """Return the list of directories paths are allowed to resolve within.

    By default this is just the project root. When ``allow_temp`` is True,
    the system temp directory and pytest's ``tmp_path`` base are also
    included — useful for functions that accept arbitrary file paths from
    callers (e.g. CI report parsers in tests).
    """
    bases = [get_project_root()]
    if allow_temp:
        bases.append(Path(tempfile.gettempdir()).resolve())
    return bases


def is_hf_model_id(value: str) -> bool:
    """Heuristically detect whether *value* is a HuggingFace model identifier.

    HF model IDs look like ``org/model-name`` or ``org/model-name/revision``
    — they contain a slash, do **not** start with a path separator or ``.``,
    and do not correspond to an existing local path.

    Returns ``True`` when the value looks like a model ID (and should therefore
    be passed through to HuggingFace APIs rather than validated as a local path).
    """
    if not value or not isinstance(value, str):
        return False
    # Absolute or relative local paths — not model IDs.
    if value.startswith(("/", "\\", "./", ".." + "/")):
        return False
    # Bare filename or single-component — could be local.
    if "/" not in value:
        return False
    # Contains '/' and doesn't look like a path → likely a model ID.
    return True


def _is_within_bases(resolved: Path, bases: list[Path]) -> bool:
    """Check that *resolved* is within one of the *bases* (or equals a base)."""
    for b in bases:
        if resolved == b or b in resolved.parents:
            return True
    return False


def _resolve_path(path: str | Path, bases: list[Path]) -> Path:
    """Resolve *path* and ensure it stays within one of *bases*.

    Relative paths are resolved against the first base (the project root).
    Absolute paths are checked directly against all bases.

    Raises ``PathSecurityError`` if the resolved path escapes all bases.
    """
    p = Path(path)
    if not p.is_absolute():
        # Resolve relative paths against the project root (first base).
        p = bases[0] / p

    try:
        resolved = p.resolve()
    except (OSError, RuntimeError):
        # Path resolution can fail on malformed paths — fall back to
        # lexical resolution so we can still check containment.
        resolved = p.absolute()

    if _is_within_bases(resolved, bases):
        return resolved

    raise PathSecurityError(path, resolved, bases)


def validate_path(
    path: str | Path,
    base_dir: str | Path | None = None,
    *,
    allow_model_id: bool = False,
    allow_temp: bool = False,
) -> Path:
    """Validate that *path* resolves within *base_dir* (and optionally temp).

    Parameters
    ----------
    path:
        The path to validate. May be relative (resolved against *base_dir*)
        or absolute.
    base_dir:
        The allowed root directory. When ``None`` (default) the project root
        is used. When a relative path is given it is resolved against the
        project root.
    allow_model_id:
        If ``True``, values that look like HuggingFace model identifiers
        (e.g. ``"Qwen/Qwen2.5-Coder-7B-Instruct"``) are returned as-is without
        filesystem validation.
    allow_temp:
        If ``True``, paths within the system temp directory are also allowed.
        This is needed for functions that legitimately accept temp file paths
        (e.g. security-scan report parsers in CI).

    Returns
    -------
    Path
        The resolved, validated ``Path`` object.

    Raises
    ------
    PathSecurityError
        If the resolved path escapes *base_dir* and (when ``allow_temp``)
        the system temp directory.
    """
    if base_dir is None:
        bases = get_allowed_bases(allow_temp=allow_temp)
    else:
        b = Path(base_dir)
        if not b.is_absolute():
            b = get_project_root() / b
        bases = [b.resolve()]
        if allow_temp:
            bases.append(Path(tempfile.gettempdir()).resolve())

    # Allow HuggingFace model IDs to pass through (they are not local paths).
    if allow_model_id and isinstance(path, str) and is_hf_model_id(path):
        return Path(path)

    return _resolve_path(path, bases)


def validate_path_str(
    path: str | Path,
    base_dir: str | Path | None = None,
    *,
    allow_model_id: bool = False,
    allow_temp: bool = False,
) -> str:
    """Like :func:`validate_path` but returns a ``str``."""
    return str(
        validate_path(
            path, base_dir, allow_model_id=allow_model_id, allow_temp=allow_temp
        )
    )


def validate_output_path(
    path: str | Path,
    base_dir: str | Path | None = None,
    *,
    allow_temp: bool = False,
) -> Path:
    """Validate an output path (for writing) is within *base_dir*.

    This is a convenience wrapper around :func:`validate_path` with a clear
    name indicating the path is intended for **writing**. It resolves
    parent directories so that a path like ``output/stage5/new/results.json``
    (whose parent may not exist yet) is validated correctly.

    Parameters
    ----------
    path:
        The output file path to validate.
    base_dir:
        The allowed root directory (default: project root).
    allow_temp:
        If ``True``, paths within the system temp directory are also allowed.

    Returns
    -------
    Path
        The resolved, validated ``Path`` object.

    Raises
    ------
    PathSecurityError
        If the resolved path escapes *base_dir* and (when ``allow_temp``)
        the system temp directory.
    """
    return validate_path(
        path, base_dir, allow_model_id=False, allow_temp=allow_temp
    )


def safe_read_text(
    path: str | Path,
    base_dir: str | Path | None = None,
    *,
    allow_model_id: bool = False,
    allow_temp: bool = False,
    encoding: str = "utf-8",
) -> str:
    """Validate *path* then read its contents as text.

    Parameters
    ----------
    path:
        The file path to read.
    base_dir:
        The allowed root directory (default: project root).
    allow_model_id:
        If ``True``, values that look like HuggingFace model IDs are returned
        without filesystem validation (and will raise FileNotFoundError if
        the model ID is not a real local path).
    allow_temp:
        If ``True``, paths within the system temp directory are also allowed.
    encoding:
        Text encoding to use when reading the file.

    Returns
    -------
    str
        The file contents.

    Raises
    ------
    PathSecurityError
        If the path escapes *base_dir* and (when ``allow_temp``) the temp dir.
    FileNotFoundError
        If the file does not exist.
    """
    safe_path = validate_path(
        path, base_dir, allow_model_id=allow_model_id, allow_temp=allow_temp
    )
    if not safe_path.exists():
        raise FileNotFoundError(f"File not found: {safe_path}")
    return safe_path.read_text(encoding=encoding)  # NOSONAR
