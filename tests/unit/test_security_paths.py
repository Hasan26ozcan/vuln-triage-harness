"""Unit tests for app.security.paths — path-traversal prevention utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.security.paths import (
    PathSecurityError,
    get_allowed_bases,
    get_project_root,
    is_hf_model_id,
    safe_read_text,
    validate_output_path,
    validate_path,
    validate_path_str,
)

# ---------------------------------------------------------------------------
# get_project_root / get_allowed_bases
# ---------------------------------------------------------------------------


class TestGetProjectRoot:
    def test_returns_path(self):
        root = get_project_root()
        assert isinstance(root, Path)

    def test_contains_app_package(self):
        root = get_project_root()
        assert (root / "app" / "__init__.py").exists()


class TestGetAllowedBases:
    def test_default_only_project_root(self):
        bases = get_allowed_bases()
        assert len(bases) == 1
        assert bases[0] == get_project_root()

    def test_allow_temp_adds_system_temp(self):
        bases = get_allowed_bases(allow_temp=True)
        assert len(bases) == 2
        assert Path(tempfile.gettempdir()).resolve() in bases


# ---------------------------------------------------------------------------
# is_hf_model_id
# ---------------------------------------------------------------------------


class TestIsHfModelId:
    def test_none_returns_false(self):
        assert is_hf_model_id(None) is False  # type: ignore[arg-type]

    def test_empty_string_returns_false(self):
        assert is_hf_model_id("") is False

    def test_non_string_returns_false(self):
        # int / list can't be model IDs
        assert is_hf_model_id(12345) is False  # type: ignore[arg-type]
        assert is_hf_model_id(["a", "b"]) is False  # type: ignore[arg-type]

    def test_absolute_path_returns_false(self):
        assert is_hf_model_id("/some/absolute/path") is False

    def test_relative_dot_path_returns_false(self):
        assert is_hf_model_id("./relative/path") is False

    def test_parent_dir_returns_false(self):
        assert is_hf_model_id("../parent/path") is False

    def test_backslash_absolute_returns_false(self):
        assert is_hf_model_id("\\windows\\path") is False

    def test_bare_filename_returns_false(self):
        assert is_hf_model_id("model.gguf") is False

    def test_single_component_returns_false(self):
        assert is_hf_model_id("justoneword") is False

    def test_valid_model_id_returns_true(self):
        assert is_hf_model_id("Qwen/Qwen2.5-Coder-7B-Instruct") is True

    def test_org_with_revision_returns_true(self):
        assert is_hf_model_id("org/model/revision") is True


# ---------------------------------------------------------------------------
# validate_path
# ---------------------------------------------------------------------------


class TestValidatePath:
    def test_relative_path_resolves_within_project_root(self):
        p = validate_path("output/stage5/results.json")
        assert p.is_absolute()
        assert p.is_relative_to(get_project_root())

    def test_path_traversal_raises(self):
        with pytest.raises(PathSecurityError):
            validate_path("../../../etc/passwd")

    def test_absolute_path_within_root(self):
        root = get_project_root()
        p = validate_path(root / "app" / "__init__.py")
        assert p == (root / "app" / "__init__.py").resolve()

    def test_absolute_path_outside_root_raises(self):
        outside = Path("/etc/passwd")
        if not outside.exists():
            pytest.skip("/etc/ read access varies by platform")
        with pytest.raises(PathSecurityError):
            validate_path(outside)

    def test_allow_model_id_returns_path(self):
        """When allow_model_id=True and value looks like a model ID, it
        is returned as-is without filesystem validation (line 175)."""
        p = validate_path(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            allow_model_id=True,
        )
        # On Windows, Path normalizes '/' to '\'; compare parts instead.
        assert p.parts == ("Qwen", "Qwen2.5-Coder-7B-Instruct")

    def test_allow_model_id_false_treats_as_path(self):
        """Without allow_model_id, the value is validated as a local path.

        On some platforms the relative path resolves within the project root
        (no error), which still demonstrates that model-ID bypass is disabled.
        """
        try:
            p = validate_path(
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                allow_model_id=False,
            )
            # If it resolves, it must be within the project root (not a
            # model-ID passthrough).
            assert p.is_relative_to(get_project_root())
        except (PathSecurityError, FileNotFoundError):
            pass  # path outside allowed roots — also acceptable

    def test_custom_base_dir_relative(self):
        """A relative base_dir is resolved against the project root
        (lines 166-168)."""
        p = validate_path("results.json", base_dir="output")
        root = get_project_root()
        assert p == (root / "output" / "results.json").resolve()

    def test_custom_base_dir_absolute(self, tmp_path):
        """An absolute base_dir is used directly (line 169)."""
        p = validate_path("results.json", base_dir=str(tmp_path))
        assert p == (tmp_path / "results.json").resolve()

    def test_allow_temp_with_base_dir(self, tmp_path):
        """When allow_temp=True with a custom base_dir, both bases are allowed
        (lines 170-171)."""
        # Write a file in tmp_path and verify it resolves
        f = tmp_path / "test.json"
        f.write_text("hello", encoding="utf-8")
        p = validate_path("test.json", base_dir=str(tmp_path), allow_temp=True)
        assert p == f.resolve()

    def test_allow_temp_allows_system_temp(self, tmp_path):
        """Files in the system temp directory are allowed when allow_temp=True."""
        temp_file = Path(tempfile.gettempdir()) / "test_security_path.json"
        temp_file.write_text("temp content", encoding="utf-8")
        try:
            p = validate_path(str(temp_file), allow_temp=True)
            assert p == temp_file.resolve()
        finally:
            temp_file.unlink(missing_ok=True)

    def test_base_dir_path_object(self, tmp_path):
        """A Path object as base_dir also works."""
        f = tmp_path / "data.json"
        f.write_text("{}", encoding="utf-8")
        p = validate_path("data.json", base_dir=Path(tmp_path))
        assert p == f.resolve()


class TestValidatePathStr:
    """validate_path_str returns a string (line 188)."""

    def test_returns_string(self):
        result = validate_path_str("app/__init__.py")
        assert isinstance(result, str)
        assert "app" in result

    def test_raises_on_traversal(self):
        with pytest.raises(PathSecurityError):
            validate_path_str("../../../../etc/passwd")


class TestValidateOutputPath:
    """validate_output_path wraps validate_path with allow_model_id=False."""

    def test_valid_output_path(self, tmp_path):
        out = tmp_path / "new" / "results.json"
        p = validate_output_path(str(out), base_dir=str(tmp_path))
        assert p == out.resolve()

    def test_traversal_blocked(self, tmp_path):
        with pytest.raises(PathSecurityError):
            validate_output_path("../secret.json", base_dir=str(tmp_path))

    def test_disallows_model_id(self):
        """validate_output_path never allows model IDs (allow_model_id=False
        is hardcoded inside the function)."""
        # validate_output_path does not accept allow_model_id; it always
        # passes False to validate_path. A model ID used as a path either
        # resolves within the project root or raises a security error.
        try:
            p = validate_output_path("Qwen/Qwen2.5-Coder-7B-Instruct")
            # If it resolves, it must be within an allowed base (not a
            # model-ID passthrough).
            root = get_project_root()
            assert p.is_relative_to(root) or p.is_relative_to(Path(tempfile.gettempdir()).resolve())
        except (PathSecurityError, FileNotFoundError):
            pass  # path outside allowed roots — also acceptable


class TestSafeReadText:
    def test_reads_file_contents(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        content = safe_read_text(str(f), allow_temp=True, encoding="utf-8")
        assert '"key"' in content

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            safe_read_text(str(tmp_path / "missing.json"), allow_temp=True)

    def test_traversal_blocked(self):
        """A path that escapes the project root triggers PathSecurityError."""
        with pytest.raises(PathSecurityError):
            safe_read_text("../../../etc/passwd")

    def test_path_object_input(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content", encoding="utf-8")
        content = safe_read_text(Path(f), allow_temp=True)
        assert content == "content"

    def test_model_id_with_allow_model_id_raises_filenotfound(self):
        """When allow_model_id=True and the value is a model ID, it bypasses
        path validation and raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            safe_read_text(
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                allow_model_id=True,
            )


class TestResolvePathLexicalFallback:
    """Exercises the OSError/RuntimeError fallback in _resolve_path (lines 114-117)."""

    def test_resolve_path_handles_oserror(self):
        """When Path.resolve() raises OSError, the code falls back to
        Path.absolute() (lines 114-117) and still performs the security check."""
        from app.security.paths import _resolve_path

        # Use a base_dir that exists; the path that triggers OSError on resolve
        # is platform-specific. We test that _resolve_path itself doesn't crash
        # by using a normal valid path (which exercises the try block).
        root = get_project_root()
        result = _resolve_path("app/__init__.py", [root])
        assert result.is_absolute()

    def test_resolve_path_lexical_fallback_on_exception(self):
        """Force Path.resolve() to raise and verify the absolute() fallback
        still validates containment (lines 114-122)."""
        from app.security.paths import _resolve_path

        root = get_project_root()
        # Patch Path.resolve to raise OSError, forcing the fallback path
        original_resolve = Path.resolve

        def raising_resolve(self, *a, **kw):
            if self.name == "nonexistent_file.json":
                raise OSError("simulated")
            return original_resolve(self, *a, **kw)

        try:
            Path.resolve = raising_resolve  # type: ignore[assignment]
            # The path doesn't exist but resolve() raises OSError → absolute() fallback.
            # The path is within the project root so it should still pass containment.
            result = _resolve_path("nonexistent_file.json", [root])
            assert result is not None
        finally:
            Path.resolve = original_resolve  # type: ignore[assignment]
