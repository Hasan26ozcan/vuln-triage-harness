"""SonarQube-style type annotation coverage checks.

Ensures that every public function and method in ``app/`` has type
annotations — SonarQube's "Fully covered" / "Commented code" quality
profile checks.  These are checked via static AST inspection so they
run fast and don't require a full mypy pass in every CI iteration.

Run directly::

    pytest tests/code_quality/test_type_coverage.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
SKIP_PREFIXES = ("__pycache__", "_", "conftest")


def _collect_python_files() -> list[Path]:
    files = []
    for p in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" not in p.parts:
            if not any(part.startswith("_") and part != "__init__.py" for part in p.parts):
                files.append(p)
    return files


def _is_public_function(node: ast.FunctionDef | ast.AsyncFunctionDef, tree: ast.Module) -> bool:
    """A function is "public" if its name doesn't start with ``_`` and it's
    not nested inside a private class."""
    if node.name.startswith("_"):
        return False
    # Check if it's a method inside a private class
    for cls in _iter_classes(tree):
        if cls.name.startswith("_"):
            for item in ast.walk(cls):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item is node:
                    return False
    return True


def _iter_classes(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            yield node


def _has_return_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a function has a return-type annotation."""
    return node.returns is not None


def _has_param_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if *all* non-self/cls params have annotations."""
    args = node.args
    # Combine pos-only, pos-or-kw, kw-only, and varargs/kwarg.
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg:
        all_args.append(args.vararg)
    if args.kwarg:
        all_args.append(args.kwarg)

    for arg in all_args:
        # Skip self/cls.
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is None:
            return False
    return True


class TestTypeAnnotationCoverage:
    """Verify that public functions and methods have full type annotations."""

    def test_all_public_functions_have_return_types(self) -> None:
        """Every public function/method must have a return-type annotation."""
        violations = []
        for pyfile in _collect_python_files():
            source = pyfile.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _is_public_function(node, tree):
                        if not _has_return_annotation(node):
                            rel = pyfile.relative_to(APP_DIR.parent)
                            violations.append(
                                f"{rel}:{node.lineno} {node.name}() — missing return annotation"
                            )
        assert not violations, (
            f"{len(violations)} public function(s) missing return-type annotations:\n"
            + "\n".join(violations)
        )

    def test_all_public_functions_have_param_types(self) -> None:
        """Every public function/method must annotate all parameters."""
        violations = []
        for pyfile in _collect_python_files():
            source = pyfile.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _is_public_function(node, tree):
                        if not _has_param_annotations(node):
                            # Find the first unannotated param.
                            all_args = (
                                list(node.args.posonlyargs)
                                + list(node.args.args)
                                + list(node.args.kwonlyargs)
                            )
                            missing = []
                            for arg in all_args:
                                if arg.arg in ("self", "cls"):
                                    continue
                                if arg.annotation is None:
                                    missing.append(arg.arg)
                            rel = pyfile.relative_to(APP_DIR.parent)
                            violations.append(
                                f"{rel}:{node.lineno} {node.name}() "
                                f"— missing param annotations: {missing}"
                            )
        assert not violations, (
            f"{len(violations)} public function(s) missing parameter annotations:\n"
            + "\n".join(violations)
        )

    def test_no_bare_dict_or_list_annotations(self) -> None:
        """Bare ``dict`` and ``list`` type annotations should use subscripted form
        (e.g. ``dict[str, Any]``) — catches incomplete type declarations that
        SonarQube would flag as 'reduce cognitive load'."""
        violations = []
        for pyfile in _collect_python_files():
            source = pyfile.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign) and node.annotation:
                    ann = node.annotation
                    # ``dict`` without subscript: ast.Name(id='dict')
                    if isinstance(ann, ast.Name) and ann.id in ("dict", "list"):
                        rel = pyfile.relative_to(APP_DIR.parent)
                        violations.append(
                            f"{rel}:{node.lineno} — bare '{ann.id}' annotation, "
                            f"use '{ann.id}[...]' instead"
                        )
        assert not violations, (
            f"{len(violations)} bare dict/list annotation(s) found:\n"
            + "\n".join(violations)
        )

    def test_type_coverage_ratio_above_threshold(self) -> None:
        """At least 90% of public functions must have both return and param annotations."""
        total_typed = 0
        total_functions = 0
        for pyfile in _collect_python_files():
            source = pyfile.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _is_public_function(node, tree):
                        total_functions += 1
                        if _has_return_annotation(node) and _has_param_annotations(node):
                            total_typed += 1
        if total_functions == 0:
            pytest.skip("No public functions found to annotate")
        ratio = total_typed / total_functions
        assert ratio >= 0.90, (
            f"Type annotation coverage is {ratio:.1%} ({total_typed}/{total_functions}) "
            f"— minimum 90% required."
        )
