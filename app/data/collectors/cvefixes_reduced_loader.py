"""Reduced-schema CVEfixes loader — works with the 3-table subset
(`fixes`, `commits`, `file_change`) when the full CVEfixes v1.0.8
schema (with `cwe_classification`, `repository`, `method_change`, `cve`)
is not available locally.

CWE classifications are provided via an external mapping file built from
the NVD data (`data/cve_cwe_mapping.json`). This mapping is generated
once by extracting the NVD CVE JSON from the CVEfixes.zip archive.

repo_name is derived from repo_url (e.g. ``https://github.com/owner/repo``
→ ``owner/repo``) since the ``repository`` table is missing.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.data.collectors.cwe_scope import CWE_IDS
from app.security.paths import validate_path


@dataclass
class RawVulnPair:
    """Same shape as cvefixes_loader.RawVulnPair."""

    cve_id: str
    cwe_id: str
    repo_name: str
    commit_sha: str
    language: str
    vulnerable_code: str
    fixed_code: str | None
    granularity: str  # "file" only — method_change table not available


def _derive_repo_name(repo_url: str) -> str:
    """Extract owner/repo from a repository URL.

    Handles GitHub, GitLab, Bitbucket, and generic URLs.
    Falls back to the last two path segments or the URL itself.
    """
    if not repo_url:
        return "unknown"
    parsed = urlparse(repo_url)
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return repo_url


_LANGUAGES = {"Python", "JavaScript", "TypeScript"}


class ReducedCveFixesLoader:
    """Load vulnerable/fixed code pairs from a reduced CVEfixes.db schema.

    Uses an external CVE-to-CWE mapping (``data/cve_cwe_mapping.json``)
    built from NVD data, since the ``cwe_classification`` table is absent.
    """

    def __init__(
        self,
        db_path: str | Path,
        cwe_mapping_path: str | Path = "data/cve_cwe_mapping.json",
    ):
        self.db_path = validate_path(db_path, allow_temp=True)
        if not self.db_path.exists():
            raise FileNotFoundError(f"CVEfixes.db not found at {self.db_path}.")
        self._cwe_mapping = self._load_cwe_mapping(cwe_mapping_path)

    def _load_cwe_mapping(self, path: str | Path) -> dict[str, str]:
        p = validate_path(path, allow_temp=True)
        if not p.exists():
            raise FileNotFoundError(
                f"CWE mapping file not found at {p}. Run the NVD extraction "
                "script to generate it from CVEfixes_v1.0.8.zip."
            )
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def _filter_sql(self) -> str:
        """Build SQL placeholder fragment for in-scope CWEs via temp table."""
        return "ic.cwe_id IN (SELECT cwe_id FROM temp.in_scope_cwe_list)"

    def _prepare_temp_tables(self, con: sqlite3.Connection) -> None:
        """Create temp tables for in-scope CVE IDs and their CWE IDs."""
        con.execute(
            "CREATE TEMP TABLE IF NOT EXISTS in_scope_cve_list "
            "(cve_id TEXT PRIMARY KEY, cwe_id TEXT)"
        )
        con.execute("CREATE TEMP TABLE IF NOT EXISTS in_scope_cwe_list (cwe_id TEXT PRIMARY KEY)")

        # Populate with in-scope entries from the NVD mapping
        rows = [
            (cve_id, cwe_id) for cve_id, cwe_id in self._cwe_mapping.items() if cwe_id in CWE_IDS
        ]
        con.executemany("INSERT OR IGNORE INTO in_scope_cve_list VALUES (?, ?)", rows)
        con.executemany(
            "INSERT OR IGNORE INTO in_scope_cwe_list VALUES (?)",
            [(cwe_id,) for cwe_id in CWE_IDS],
        )
        con.commit()

    def load_pairs(self, languages: set[str] | None = None) -> list[RawVulnPair]:
        """Load all in-scope pairs at file granularity.

        Since ``method_change`` is unavailable, we use ``file_change.code_before``
        / ``code_change`` directly (file-level pairs).
        """
        langs = languages or _LANGUAGES
        lang_placeholders = ",".join("?" for _ in langs)
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        try:
            self._prepare_temp_tables(con)

            # `lang_placeholders` only ever contains "?" tokens (one per entry
            # in `langs`), never external data — actual values are bound
            # through `params` via sqlite3's parameterized execute() below,
            # never string-interpolated. Same pattern as cvefixes_loader.py.
            query = f"""
                SELECT
                    f.cve_id            AS cve_id,
                    ic.cwe_id           AS cwe_id,
                    f.repo_url          AS repo_url,
                    f.hash              AS commit_sha,
                    fc.programming_language AS language,
                    fc.code_before      AS code_before,
                    fc.code_after       AS code_after,
                    fc.file_change_id   AS file_change_id
                FROM in_scope_cve_list ic
                JOIN fixes f            ON f.cve_id = ic.cve_id
                JOIN file_change fc     ON fc.hash = f.hash
                WHERE ic.cwe_id IN (SELECT cwe_id FROM in_scope_cwe_list)
                  AND fc.programming_language IN ({lang_placeholders})
                  AND fc.code_before IS NOT NULL
                  AND fc.code_after IS NOT NULL
                  AND LENGTH(fc.code_before) > 50
            """  # nosec
            params = list(langs)
            rows = con.execute(query, params).fetchall()

            results: list[RawVulnPair] = []
            for row in rows:
                results.append(
                    RawVulnPair(
                        cve_id=row["cve_id"],
                        cwe_id=row["cwe_id"],
                        repo_name=_derive_repo_name(row["repo_url"]),
                        commit_sha=row["commit_sha"],
                        language=(row["language"] or "").lower(),
                        vulnerable_code=row["code_before"],
                        fixed_code=row["code_after"],
                        granularity="file",
                    )
                )
            return results
        finally:
            con.close()
