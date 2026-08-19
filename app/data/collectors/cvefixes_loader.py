"""Loads raw vulnerable/fixed code pairs out of a CVEfixes.db SQLite file.

Schema reference (CVEfixes v1.0.8, secureIT-project/CVEfixes,
Doc/DataDictionary.md — verified against the project's own docs, not
guessed): tables `cve`, `fixes`, `repository`, `commits`, `file_change`,
`method_change`, `cwe`, `cwe_classification`.

Key relationships used here:
    cwe_classification(cve_id, cwe_id)  -- filter to our CWE_SCOPE
    fixes(cve_id, hash, repo_url)       -- CVE -> fixing commit
    repository(repo_url, repo_name)     -- commit -> repo (leakage-safe split key)
    file_change(hash, file_change_id, programming_language, code_before, code_after)
    method_change(file_change_id, name, code, before_change)  -- function-level pairs

We prefer method-level (function) granularity over whole-file diffs: it's
what Stage 3's token budget wants, and it avoids dragging unrelated code
into `vulnerable_code`/`fixed_code`. A method_change row with
`before_change=1` pairs with the row of the same `name` in the same
`file_change_id` where `before_change=0`. If a file has no method_change
rows (e.g. the fix wasn't inside a function CVEfixes could parse), we fall
back to the file-level `code_before`/`code_after` from `file_change` — the
caller (pipeline.py) is expected to enforce a size cutoff before those go
into `InstructionExample` in Stage 3.

The CVEfixes.db file itself is not bundled with this repo (it's a
multi-GB Zenodo download, see README: https://zenodo.org/records/13118970)
— this module only reads a local copy the caller has already downloaded.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.data.collectors.cwe_scope import CWE_IDS


@dataclass
class RawVulnPair:
    """One vulnerable/fixed code pair, still unenriched (no severity,
    no static findings, no NVD description) — that happens in pipeline.py.
    """

    cve_id: str
    cwe_id: str
    repo_name: str
    commit_sha: str
    language: str
    vulnerable_code: str
    fixed_code: str | None
    granularity: str  # "method" or "file"


class CveFixesLoader:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"CVEfixes.db not found at {self.db_path}. Download it from "
                "https://zenodo.org/records/13118970 first — it is not "
                "checked into this repo (multi-GB dataset)."
            )

    def load_pairs(self, languages: set[str] | None = None) -> list[RawVulnPair]:
        """Load all in-scope (CWE_SCOPE) vulnerable/fixed pairs, at method
        granularity where possible, falling back to file granularity.
        """
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        try:
            pairs = self._load_method_level(con, languages)
            covered_file_change_ids = {p[0] for p in pairs}
            pairs_file_level = self._load_file_level_fallback(
                con, languages, exclude_file_change_ids=covered_file_change_ids
            )
            return [p[1] for p in pairs] + pairs_file_level
        finally:
            con.close()

    def _in_scope_cwe_filter(self) -> str:
        placeholders = ",".join("?" for _ in CWE_IDS)
        return placeholders

    def _load_method_level(
        self, con: sqlite3.Connection, languages: set[str] | None
    ) -> list[tuple[str, RawVulnPair]]:
        query_template = """
            SELECT
                f.cve_id            AS cve_id,
                cc.cwe_id           AS cwe_id,
                r.repo_name         AS repo_name,
                fx.hash             AS commit_sha,
                fc.programming_language AS language,
                fc.file_change_id   AS file_change_id,
                mc_before.name      AS method_name,
                mc_before.code      AS code_before,
                mc_after.code       AS code_after
            FROM cwe_classification cc
            JOIN fixes fx            ON fx.cve_id = cc.cve_id
            JOIN repository r        ON r.repo_url = fx.repo_url
            JOIN file_change fc      ON fc.hash = fx.hash
            JOIN method_change mc_before
                ON mc_before.file_change_id = fc.file_change_id
                AND mc_before.before_change = 1
            JOIN method_change mc_after
                ON mc_after.file_change_id = fc.file_change_id
                AND mc_after.before_change = 0
                AND mc_after.name = mc_before.name
            JOIN cve f ON f.cve_id = fx.cve_id
            WHERE cc.cwe_id IN (__CWE_PLACEHOLDERS__)
        """
        # Only a fixed "?,?,?..." placeholder count (derived from the
        # hardcoded CWE_IDS constant, never user input) is substituted here.
        # Actual values are bound through `params` via sqlite3's
        # parameterized execute() below, never string-interpolated.
        query = query_template.replace("__CWE_PLACEHOLDERS__", self._in_scope_cwe_filter())
        params: list[str] = list(CWE_IDS)
        if languages:
            placeholders = ",".join("?" for _ in languages)
            query += f" AND fc.programming_language IN ({placeholders})"
            params += list(languages)

        rows = con.execute(query, params).fetchall()
        results: list[tuple[str, RawVulnPair]] = []
        for row in rows:
            pair = RawVulnPair(
                cve_id=row["cve_id"],
                cwe_id=row["cwe_id"],
                repo_name=row["repo_name"],
                commit_sha=row["commit_sha"],
                language=(row["language"] or "").lower(),
                vulnerable_code=row["code_before"],
                fixed_code=row["code_after"],
                granularity="method",
            )
            results.append((row["file_change_id"], pair))
        return results

    def _load_file_level_fallback(
        self,
        con: sqlite3.Connection,
        languages: set[str] | None,
        exclude_file_change_ids: set[str],
    ) -> list[RawVulnPair]:
        query_template = """
            SELECT
                fx.cve_id           AS cve_id,
                cc.cwe_id           AS cwe_id,
                r.repo_name         AS repo_name,
                fx.hash             AS commit_sha,
                fc.programming_language AS language,
                fc.file_change_id   AS file_change_id,
                fc.code_before      AS code_before,
                fc.code_after       AS code_after
            FROM cwe_classification cc
            JOIN fixes fx            ON fx.cve_id = cc.cve_id
            JOIN repository r        ON r.repo_url = fx.repo_url
            JOIN file_change fc      ON fc.hash = fx.hash
            WHERE cc.cwe_id IN (__CWE_PLACEHOLDERS__)
                AND fc.code_before IS NOT NULL
        """
        # See justification in _load_method_level above — same
        # fixed-placeholder-count pattern, values bound via `params`.
        query = query_template.replace("__CWE_PLACEHOLDERS__", self._in_scope_cwe_filter())
        params: list[str] = list(CWE_IDS)
        if languages:
            placeholders = ",".join("?" for _ in languages)
            query += f" AND fc.programming_language IN ({placeholders})"
            params += list(languages)

        rows = con.execute(query, params).fetchall()
        results = []
        for row in rows:
            if row["file_change_id"] in exclude_file_change_ids:
                continue  # already covered at method granularity, avoid duplicates
            results.append(
                RawVulnPair(
                    cve_id=row["cve_id"],
                    cwe_id=row["cwe_id"],
                    repo_name=row["repo_name"],
                    commit_sha=row["commit_sha"],
                    language=(row["language"] or "").lower(),
                    vulnerable_code=row["code_before"],
                    fixed_code=row["code_after"],
                    granularity="file",
                )
            )
        return results
