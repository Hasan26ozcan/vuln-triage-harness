"""Stage 9 — serve the quantized GGUF model and make a real request.

This script:
1. Starts the llama-server.exe binary (or llama-cpp-python, or transformers)
   with the Stage 9 GGUF checkpoint
2. Sends a real vulnerability-analysis request
3. Parses the model response and saves results to output/stage9/

Usage::

    python scripts/run_stage9_serve.py --model output/stage9/tuned_model.gguf

Backend selection
-----------------
Use ``--backend auto`` (default) to try backends in order:

1. ``llama-server`` — the bundled ``llama-server.exe`` binary (HTTP API)
2. ``llama.cpp`` — the ``llama-cpp-python`` bindings (requires pip install)
3. ``transformers`` — HuggingFace ``transformers`` + ``torch`` (GPU-capable
   fallback; uses the HF-format model directory, not the GGUF file)

Use ``--backend llama-server``, ``--backend llama.cpp``, or
``--backend transformers`` to force a specific backend.

For the ``transformers`` backend, a HuggingFace model directory is needed.
If ``--hf-model-dir`` is not specified, the script auto-derives it from
the GGUF model path by looking for a sibling directory containing
``config.json`` + ``model.safetensors``.
"""

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Ensure the project root is on sys.path when run as a script.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.security.paths import validate_output_path, validate_path  # noqa: E402

# Path to the llama-server.exe binary bundled in tools/llama-cpp/.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
_LLAMA_SERVER_EXE = os.path.join(_REPO_ROOT, "tools", "llama-cpp", "llama-server.exe")

# A representative vulnerability sample for testing the serving layer.
SAMPLE_REQUEST = {
    "sample_id": "stage9-serve-test",
    "vulnerable_code": (
        "import sqlite3\n"
        "def get_user(user_id):\n"
        "    conn = sqlite3.connect('users.db')\n"
        "    return cursor.fetchone()\n"
        "conn.close()\n"
    ),
    "language": "python",
    "cwe_id": "CWE-89",
    "severity": "high",
    "description": "SQL injection via unsanitized user input in f-string query.",
    "static_findings": [],
}


# Re-inject the full vulnerable code (kept short above for brevity).
SAMPLE_REQUEST["vulnerable_code"] = (
    "import sqlite3\n"
    "def get_user(user_id):\n"
    "    conn = sqlite3.connect('users.db')\n"
    "    cursor = conn.cursor()\n"
    "    # VULNERABILITY: direct string interpolation into SQL query\n"
    '    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")\n'
    "    return cursor.fetchone()\n"
    "conn.close()\n"
)


def _try_backend(prompt, model_path, backend_name, port, hf_model_dir):
    """Attempt to create and use *backend_name*; return (backend, response, type, binary).

    Returns ``(None, None, None, None)`` if the backend cannot be initialised
    or fails during generation.
    """
    from app.serving.backends import (
        LlamaCppBackend,
        LlamaServerBackend,
        TransformersBackend,
        _find_hf_model_dir,
    )

    if backend_name == "llama-server":
        if not os.path.exists(_LLAMA_SERVER_EXE):
            print(f"  llama-server.exe not found at {_LLAMA_SERVER_EXE}", file=sys.stderr)
            return None, None, None, None
        print(f"  Trying llama-server on port {port} ...")
        backend = LlamaServerBackend(
            model_path=model_path,
            server_binary=_LLAMA_SERVER_EXE,
            host="127.0.0.1",
            port=port,
            num_threads=4,
            num_ctx=4096,
            temperature=0.2,
            max_new_tokens=512,
            request_timeout=60.0,
        )
        binary = _LLAMA_SERVER_EXE
        bt = "llama-server"
    elif backend_name == "llama.cpp":
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            print("  llama-cpp-python not installed — skipping.", file=sys.stderr)
            return None, None, None, None
        print("  Trying llama-cpp-python backend ...")
        backend = LlamaCppBackend(
            model_path=model_path,
            num_ctx=4096,
            num_threads=4,
            n_gpu_layers=0,
            f16_kv=True,
            temperature=0.2,
            max_new_tokens=512,
        )
        binary = "(llama-cpp-python in-process)"
        bt = "llama.cpp"
    elif backend_name == "transformers":
        hf_dir = hf_model_dir or _find_hf_model_dir(model_path)
        if hf_dir is None:
            print(
                f"  No HuggingFace model dir found for {model_path} "
                "(pass --hf-model-dir to specify one)",
                file=sys.stderr,
            )
            return None, None, None, None
        if not os.path.isdir(hf_dir):
            print(f"  HF model dir not found: {hf_dir}", file=sys.stderr)
            return None, None, None, None
        print(f"  Trying transformers backend with {hf_dir} ...")
        backend = TransformersBackend(
            model_dir=hf_dir,
            num_ctx=4096,
            num_threads=4,
            temperature=0.2,
            max_new_tokens=512,
        )
        binary = f"(transformers: {hf_dir})"
        bt = "transformers"
    else:
        return None, None, None, None

    # Try to generate — this is where the server binary / llama-cpp loads.
    try:
        raw_response = backend.generate(prompt)
        return backend, raw_response, bt, binary
    except Exception as exc:
        print(f"  Backend '{backend_name}' generate() failed: {exc}", file=sys.stderr)
        if hasattr(backend, "close"):
            try:
                backend.close()
            except Exception:  # nosec B110 — best-effort close during cleanup
                pass
        return None, None, None, None


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Stage 9: serve GGUF model and make a real request")
    ap.add_argument(
        "--model",
        default="output/stage8/qwen2_gguf_f32.gguf",
        help="Path to the GGUF checkpoint from Stage 8.",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8082,
        help="Port for the llama-server subprocess (default: 8082).",
    )
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "llama-server", "llama.cpp", "transformers"],
        help="Serving backend to use.",
    )
    ap.add_argument(
        "--hf-model-dir",
        default=None,
        help="HuggingFace model directory (for the transformers backend; "
        "auto-derived from --model if omitted).",
    )
    args = ap.parse_args()

    safe_model = validate_path(args.model, allow_temp=True)
    model_path = str(safe_model)
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    # Validate HF model dir if provided (CLI arg — potential traversal vector).
    safe_hf_dir = None
    if args.hf_model_dir:
        safe_hf_dir = str(validate_path(args.hf_model_dir, allow_temp=True))

    # Build the prompt using the same Stage 4 zero-shot prompt template.
    from app.evaluation.prompt import build_zero_shot_prompt
    from app.schemas.vuln import VulnSample

    vuln_sample = VulnSample(
        id=SAMPLE_REQUEST["sample_id"],
        source="synthetic_injected",
        repo_name="vuln-triage-harness",
        cwe_id=SAMPLE_REQUEST["cwe_id"],
        severity=SAMPLE_REQUEST["severity"],
        language=SAMPLE_REQUEST["language"],
        vulnerable_code=SAMPLE_REQUEST["vulnerable_code"],
        description=SAMPLE_REQUEST["description"],
        static_findings=[],
    )
    prompt = build_zero_shot_prompt(vuln_sample)
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Prompt (first 300 chars):\n{prompt[:300]}...")

    # ------------------------------------------------------------------
    # Backend selection — try each backend in order until one succeeds.
    # ------------------------------------------------------------------
    if args.backend == "auto":
        candidates = ["llama-server", "llama.cpp", "transformers"]
    else:
        candidates = [args.backend]

    backend = None
    raw_response = None
    backend_type_used = None
    server_binary = None

    for name in candidates:
        print(f"\n=== Attempting backend: {name} ===", flush=True)
        t0 = time.perf_counter()
        backend, raw_response, bt, binary = _try_backend(
            prompt, model_path, name, args.port, safe_hf_dir
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if backend is not None:
            backend_type_used = bt
            server_binary = binary
            print(f"  Backend '{name}' succeeded in {elapsed_ms} ms")
            break
        print(f"  Backend '{name}' failed (total {elapsed_ms} ms), trying next ...")

    if backend is None:
        print("\nERROR: All requested backends failed.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Parse & save
    # ------------------------------------------------------------------
    output_dir = validate_output_path("output/stage9", allow_temp=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\nResponse received (backend: {backend_type_used})")
        print(f"Response (first 500 chars):\n{raw_response[:500]}...")

        from app.evaluation.parser import parse_prediction

        result = parse_prediction(
            raw_response,
            sample_id=SAMPLE_REQUEST["sample_id"],
            run_id="stage9_serve",
        )

        if hasattr(result, "predicted_cwe"):
            print(f"\nParsed: CWE={result.predicted_cwe}, Severity={result.predicted_severity}")
            parsed = True
        else:
            print(f"\nParse error: {result.reason}")
            parsed = False

        serve_result = {
            "run_id": "stage9_serve",
            "model_path": model_path,
            "server_binary": server_binary,
            "port": args.port if backend_type_used == "llama-server" else None,
            "backend": backend_type_used,
            "timestamp": datetime.now(UTC).isoformat(),
            "sample": SAMPLE_REQUEST,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed": parsed,
            "predicted_cwe": result.predicted_cwe if hasattr(result, "predicted_cwe") else None,
            "predicted_severity": (
                result.predicted_severity if hasattr(result, "predicted_cwe") else None
            ),
            "parse_error": result.reason if not hasattr(result, "predicted_cwe") else None,
            "model_info": backend.model_info,
        }

        output_file = validate_output_path(output_dir / "serve_result.json", allow_temp=True)
        output_file.write_text(json.dumps(serve_result, indent=2, default=str), encoding="utf-8")
        print(f"\nResults saved to {output_file}")

        print("\n=== Stage 9 Serving Summary ===")
        print(f"  Backend:   {backend_type_used}")
        print(f"  Model:     {model_path}")
        if backend_type_used == "llama-server":
            print(f"  Port:      {args.port}")
        print(f"  Parsed:    {parsed}")
        if parsed:
            print(f"  CWE:       {result.predicted_cwe}")
            print(f"  Severity:  {result.predicted_severity}")

    finally:
        if hasattr(backend, "close"):
            backend.close()
            print("\nBackend shut down.")
        else:
            print("\nBackend shut down.")

    print("\n=== Stage 9 Complete ===")


if __name__ == "__main__":
    main()
