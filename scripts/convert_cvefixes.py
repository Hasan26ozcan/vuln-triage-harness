#!/usr/bin/env python
"""Convert the 12.6-GB PostgreSQL->SQLite SQL dump inside the zip into a
SQLite database that ``CveFixesLoader`` can consume.

Strategy
--------
Stream-decompress the gzipped SQL from inside the zip, accumulate text into
a buffer, and find safe statement boundaries (semicolon *outside* a
single-quoted string) using a fast state machine that jumps to `'` and `;`
characters via ``str.find`` — no per-character Python loop.

The string-literal-aware splitter handles ``''`` (doubled single quote =
escaped quote inside a string), which appears in code snippets stored in
``file_change.code_before`` etc.

Usage::

    python scripts/convert_cvefixes.py \
        --zip data/downloads/CVEfixes_v1.0.8.zip \
        --sql CVEfixes_v1.0.8/Data/CVEfixes_v1.0.8.sql.gz \
        --out data/cvefixes_db/CVEfixes.db
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

# Ensure project root is on sys.path for security utilities.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.security.paths import validate_output_path, validate_path  # noqa: E402

CHUNK_SIZE = 256 * 1024 * 1024  # 256 MB


def find_last_safe_semicolon(text: str) -> int:
    """Index of the last ``;`` outside a single-quoted string, or -1.

    Uses str.find() to jump directly to quote/semicolon characters, avoiding
    a slow per-character Python loop.  Handles ``''`` (SQL escaped quote).
    """
    last = -1
    in_str = False
    i = 0
    n = len(text)
    while i < n:
        if in_str:
            # Find next quote
            q = text.find("'", i)
            if q == -1:
                break
            # Check for escaped quote ''
            if q + 1 < n and text[q + 1] == "'":
                i = q + 2
                continue
            in_str = False
            i = q + 1
        else:
            # Find next quote or semicolon
            q = text.find("'", i)
            s = text.find(";", i)
            if q == -1 and s == -1:
                break
            if q != -1 and (s == -1 or q < s):
                in_str = True
                i = q + 1
            elif s != -1:
                last = s
                i = s + 1
            else:
                break
    return last


def convert(zip_path: str, sql_name: str, out_path: str) -> None:
    safe_zip = validate_path(zip_path, allow_temp=True)
    safe_out = validate_output_path(out_path, allow_temp=True)
    os.makedirs(os.path.dirname(safe_out) or ".", exist_ok=True)  # NOSONAR
    if os.path.exists(safe_out):
        os.remove(safe_out)  # NOSONAR

    # safe_out is already validated by validate_output_path — it resolves
    # strictly within the project root or temp dir, so no URI-style DB
    # connection strings (e.g. "file:...", ":memory:") can be injected
    # via CLI args (CWE-89 / connection injection mitigation).
    conn = sqlite3.connect(str(safe_out))  # NOSONAR
    conn.execute("PRAGMA journal_mode=MEMORY;")  # faster than WAL for bulk load
    conn.execute("PRAGMA synchronous=OFF;")

    start = time.monotonic()
    carry = ""
    total_stmts = 0
    batches = 0

    with zipfile.ZipFile(safe_zip) as zf, zf.open(sql_name) as raw:
        with gzip.GzipFile(fileobj=io.BufferedReader(raw)) as gz:  # type: ignore[type-var]
            while True:
                chunk = gz.read(CHUNK_SIZE)
                if not chunk:
                    break
                carry += chunk.decode("utf-8", errors="replace")

                split = find_last_safe_semicolon(carry)
                if split < 0:
                    continue

                complete = carry[: split + 1]
                carry = carry[split + 1 :]

                conn.executescript(complete)
                total_stmts += complete.count(";")
                batches += 1
                elapsed = time.monotonic() - start
                print(
                    f"  [{elapsed:7.1f}s] batch {batches}, ~{total_stmts / 1_000_000:.1f}M stmts",
                    flush=True,
                )

    # Flush remainder
    if carry.strip():
        conn.executescript(carry)
        total_stmts += carry.count(";")

    conn.close()
    elapsed = time.monotonic() - start
    print(
        f"\nDone — {total_stmts:,} statements in {batches} batches ({elapsed:.1f}s)",
        flush=True,
    )
    print(f"DB size: {os.path.getsize(safe_out) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--sql", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convert(args.zip, args.sql, args.out)
