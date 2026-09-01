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

# Backend names — shared between selection and display (SonarQube S1132).
_BACKEND_LLAMA_SERVER = "llama-server"
_BACKEND_LLAMA_CPP = "llama.cpp"
_BACKEND_TRANSFORMERS = "transformers"

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


def _create_backend(backend_name, model_path, port, hf_model_dir):
    """Create and return ``(backend, binary, backend_type)`` for *backend_name*.

    Returns ``None`` if the backend cannot be initialised (missing binary,
    missing dependency, or missing HF model dir).
    """
    from app.serving.backends import (
        LlamaCppBackend,
        LlamaServerBackend,
        TransformersBackend,
        _find_hf_model_dir,
    )

    if backend_name == _BACKEND_LLAMA_SERVER:
        if not os.path.exists(_LLAMA_SERVER_EXE):
            print(f"  llama-server.exe not found at {_LLAMA_SERVER_EXE}", file=sys.stderr)
            return None
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
        return backend, _LLAMA_SERVER_EXE, _BACKEND_LLAMA_SERVER
    elif backend_name == _BACKEND_LLAMA_CPP:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            print("  llama-cpp-python not installed — skipping.", file=sys.stderr)
            return None
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
        return backend, "(llama-cpp-python in-process)", _BACKEND_LLAMA_CPP
    elif backend_name == _BACKEND_TRANSFORMERS:
        hf_dir = hf_model_dir or _find_hf_model_dir(model_path)
        if hf_dir is None:
            print(
                f"  No HuggingFace model dir found for {model_path} "
                "(pass --hf-model-dir to specify one)",
                file=sys.stderr,
            )
            return None
        if not os.path.isdir(hf_dir):
            print(f"  HF model dir not found: {hf_dir}", file=sys.stderr)
            return None
        print(f"  Trying transformers backend with {hf_dir} ...")
        backend = TransformersBackend(
            model_dir=hf_dir,
            num_ctx=4096,
            num_threads=4,
            temperature=0.2,
            max_new_tokens=512,
        )
        return backend, f"(transformers: {hf_dir})", _BACKEND_TRANSFORMERS
    return None


def _try_backend(prompt, model_path, backend_name, port, hf_model_dir):
    """Attempt to create and use *backend_name*; return (backend, response, type, binary).

    Returns ``(None, None, None, None)`` if the backend cannot be initialised
    or fails during generation.
    """
    created = _create_backend(backend_name, model_path, port, hf_model_dir)
    if created is None:
        return None, None, None, None
    backend, binary, bt = created

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


def _build_args():
    """Parse CLI arguments for Stage 9 serving."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Stage 9: serve GGUF model and make a real request"
    )
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
        choices=["auto", _BACKEND_LLAMA_SERVER, _BACKEND_LLAMA_CPP, _BACKEND_TRANSFORMERS],
        help="Serving backend to use.",
    )
    ap.add_argument(
        "--hf-model-dir",
        default=None,
        help="HuggingFace model directory (for the transformers backend; "
        "auto-derived from --model if omitted).",
    )
    return ap.parse_args()


def _select_backend(prompt, model_path, args, safe_hf_dir):
    """Try candidate backends in order; return (backend, response, type, binary)."""
    if args.backend == "auto":
        candidates = [_BACKEND_LLAMA_SERVER, _BACKEND_LLAMA_CPP, _BACKEND_TRANSFORMERS]
    else:
        candidates = [args.backend]

    for name in candidates:
        print(f"\n=== Attempting backend: {name} ===", flush=True)
        t0 = time.perf_counter()
        backend, raw_response, bt, binary = _try_backend(
            prompt, model_path, name, args.port, safe_hf_dir
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if backend is not None:
            print(f"  Backend '{name}' succeeded in {elapsed_ms} ms")
            return backend, raw_response, bt, binary
        print(f"  Backend '{name}' failed (total {elapsed_ms} ms), trying next ...")

    return None, None, None, None


def _parse_and_save(
    backend, raw_response, backend_type_used, server_binary, model_path, args, prompt
):
    """Parse model response and persist results to disk. Returns ``parsed`` bool."""
    from app.evaluation.parser import parse_prediction

    print(f"\nResponse received (backend: {backend_type_used})")
    print(f"Response (first 500 chars):\n{raw_response[:500]}...")

    result = parse_prediction(
        raw_response,
        sample_id=SAMPLE_REQUEST["sample_id"],
        run_id="stage9_serve",
    )

    has_cwe = hasattr(result, "predicted_cwe")
    parsed = has_cwe
    if has_cwe:
        print(f"\nParsed: CWE={result.predicted_cwe}, Severity={result.predicted_severity}")
    else:
        print(f"\nParse error: {result.reason}")

    serve_result = {
        "run_id": "stage9_serve",
        "model_path": model_path,
        "server_binary": server_binary,
        "port": args.port if backend_type_used == _BACKEND_LLAMA_SERVER else None,
        "backend": backend_type_used,
        "timestamp": datetime.now(UTC).isoformat(),
        "sample": SAMPLE_REQUEST,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed": parsed,
        "predicted_cwe": result.predicted_cwe if has_cwe else None,
        "predicted_severity": result.predicted_severity if has_cwe else None,
        "parse_error": result.reason if not has_cwe else None,
        "model_info": backend.model_info,
    }

    output_dir = validate_output_path("output/stage9", allow_temp=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = validate_output_path(output_dir / "serve_result.json", allow_temp=True)
    content = json.dumps(serve_result, indent=2, default=str)
    output_file.write_text(content, encoding="utf-8")  # NOSONAR
    print(f"\nResults saved to {output_file}")

    print("\n=== Stage 9 Serving Summary ===")
    print(f"  Backend:   {backend_type_used}")
    print(f"  Model:     {model_path}")
    if backend_type_used == _BACKEND_LLAMA_SERVER:
        print(f"  Port:      {args.port}")
    print(f"  Parsed:    {parsed}")
    if parsed:
        print(f"  CWE:       {result.predicted_cwe}")
        print(f"  Severity:  {result.predicted_severity}")
    return parsed


def main():
    args = _build_args()

    safe_model = validate_path(args.model, allow_temp=True)
    model_path = str(safe_model)
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

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

    backend, raw_response, backend_type_used, server_binary = _select_backend(
        prompt, model_path, args, safe_hf_dir
    )
    if backend is None:
        print("\nERROR: All requested backends failed.", file=sys.stderr)
        sys.exit(1)

    try:
        _parse_and_save(
            backend, raw_response, backend_type_used, server_binary,
            model_path, args, prompt,
        )
    finally:
        if hasattr(backend, "close"):
            backend.close()
            print("\nBackend shut down.")
        else:
            print("\nBackend shut down.")

    print("\n=== Stage 9 Complete ===")


if __name__ == "__main__":
    main()
