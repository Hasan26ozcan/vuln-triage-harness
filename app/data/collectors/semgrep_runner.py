"""Runs Semgrep against a single code snippet (Stage 1, step 4).

Semgrep needs a file on disk (it doesn't take stdin for rule scanning), so
we write the snippet to a temp file with the right extension, run the CLI
with `--json`, and parse results into `StaticFinding`. If Semgrep finds
nothing, we deliberately return an empty list rather than raising — a
sample without static-analysis signal is still a valid, informative
`VulnSample` (per roadmap: "modelin statik analiz olmadan da tespit
edebiliyor mu" is itself a distinction worth keeping).
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 -- used only to invoke the local `semgrep` CLI, argv is a fixed list, shell=False
import tempfile
from pathlib import Path

from app.schemas.vuln import StaticFinding

_EXTENSION_BY_LANGUAGE = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "go": ".go",
    "java": ".java",
    "c": ".c",
    "cpp": ".cpp",
}

_RULES_DIR = Path(__file__).parent / "rules"

# Deliberately NOT "auto": that config pulls rulesets from semgrep.dev at
# scan time, which makes every run non-reproducible (rules can change
# under you) and fails outright in network-restricted environments (CI
# runners, air-gapped boxes — see Stage 9). Instead we ship a small,
# version-controlled rule pack per language in rules/, scoped to exactly
# the CWE classes in CWE_SCOPE. Anyone re-running this pipeline a year
# from now gets the same findings we got today.
_DEFAULT_CONFIG_BY_LANGUAGE = {
    "python": str(_RULES_DIR / "python.yaml"),
    "javascript": str(_RULES_DIR / "javascript.yaml"),
    "typescript": str(_RULES_DIR / "javascript.yaml"),
}


class SemgrepUnavailableError(RuntimeError):
    """Raised when the `semgrep` binary isn't on PATH."""


def run_semgrep(
    code: str, language: str, config: str | None = None, timeout: int = 60
) -> list[StaticFinding]:
    extension = _EXTENSION_BY_LANGUAGE.get(language)
    if extension is None:
        raise ValueError(f"Unsupported language for Semgrep: {language!r}")

    if config is None:
        config = _DEFAULT_CONFIG_BY_LANGUAGE.get(language)
        if config is None:
            raise ValueError(
                f"No bundled Semgrep rule pack for language {language!r}. "
                "Pass config= explicitly or add rules/{language}.yaml."
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        code_path = Path(tmp_dir) / f"snippet{extension}"
        code_path.write_text(code, encoding="utf-8")

        semgrep_bin = shutil.which("semgrep")
        if semgrep_bin is None:
            raise SemgrepUnavailableError(
                "semgrep is not installed / not on PATH. Install it with "
                "`pip install semgrep` or `pip install -e '.[data]'`."
            )

        # argv is a fixed list built entirely from this function's own
        # parameters (an absolute binary path resolved via shutil.which, a
        # bundled/local config path, and a temp file we just wrote);
        # shell=False, no shell metacharacters ever reach it.
        result = subprocess.run(  # nosec B603
            [semgrep_bin, "--config", config, "--json", "--quiet", str(code_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode not in (0, 1):
            # 0 = clean run, no findings, 1 = clean run, findings present.
            # Anything else is a real Semgrep error (bad config, crash, etc).
            raise RuntimeError(f"semgrep failed (exit {result.returncode}): {result.stderr[:500]}")

        payload = json.loads(result.stdout)
        return [_to_static_finding(r) for r in payload.get("results", [])]


def _to_static_finding(raw: dict) -> StaticFinding:
    start = raw.get("start", {}).get("line", 0)
    end = raw.get("end", {}).get("line", start)
    message = raw.get("extra", {}).get("message", "")
    return StaticFinding(
        tool="semgrep",
        rule_id=raw.get("check_id", "unknown"),
        message=message,
        line_range=(start, end),
    )
