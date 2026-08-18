"""Stage 9 — serve the quantized GGUF model via llama-server.exe and make a real request.

This script:
1. Starts the llama-server.exe binary with the Stage 8 GGUF checkpoint
2. Sends a real vulnerability-analysis request via HTTP /completion
3. Parses the model response and saves results to output/stage9/

Usage::

    python scripts/run_stage9_serve.py --model output/stage8/gguf_bits4.gguf
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


# Path to the llama-server.exe binary bundled in tools/llama-cpp/.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
_LLAMA_SERVER_EXE = os.path.join(
    _REPO_ROOT, "tools", "llama-cpp", "llama-server.exe"
)

# A representative vulnerability sample for testing the serving layer.
SAMPLE_REQUEST = {
    "sample_id": "stage9-serve-test",
    "vulnerable_code": (
        "import sqlite3\n"
        "def get_user(user_id):\n"
        "    conn = sqlite3.connect('users.db')\n"
        "    cursor = conn.cursor()\n"
        "    # VULNERABILITY: direct string interpolation into SQL query\n"
        "    cursor.execute(f\"SELECT * FROM users WHERE id = {user_id}\")\n"
        "    return cursor.fetchone()\n"
        "conn.close()\n"
    ),
    "language": "python",
    "cwe_id": "CWE-89",
    "severity": "high",
    "description": "SQL injection via unsanitized user input in f-string query.",
    "static_findings": [],
}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Stage 9: serve GGUF model and make a real request")
    ap.add_argument(
        "--model",
        default="output/stage8/gguf_bits4.gguf",
        help="Path to the quantized GGUF checkpoint from Stage 8.",
    )
    ap.add_argument(
        "--port", type=int, default=8082,
        help="Port for the llama-server subprocess (default: 8082 to avoid conflicts).",
    )
    args = ap.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"ERROR: GGUF model not found at {model_path}", file=sys.stderr)
        print("Run Stage 8 quantization first:", file=sys.stderr)
        print(
            "  python -m app.quantization.cli run "
            "--checkpoint ./output/stage5/dpo/final_checkpoint "
            "--method gguf --bits 4",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(_LLAMA_SERVER_EXE):
        print(f"ERROR: llama-server.exe not found at {_LLAMA_SERVER_EXE}", file=sys.stderr)
        sys.exit(1)

    # Build the prompt using the same Stage 4 zero-shot prompt template.
    from app.evaluation.prompt import build_zero_shot_prompt
    from app.schemas.vuln import VulnSample
    from app.serving.backends import LlamaServerBackend

    vuln_sample = VulnSample(
        id=SAMPLE_REQUEST["sample_id"],
        source="serving-test",
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

    # Start the llama-server subprocess.
    print(f"\n=== Starting llama-server on port {args.port} ===")
    backend = LlamaServerBackend(
        model_path=model_path,
        server_binary=_LLAMA_SERVER_EXE,
        host="127.0.0.1",
        port=args.port,
        num_threads=4,
        num_ctx=4096,
        temperature=0.2,
        max_new_tokens=512,
        request_timeout=60.0,
    )

    output_dir = Path("output/stage9")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Make a real inference request.
        print("\n=== Sending real request to llama-server ===", flush=True)
        t0 = time.perf_counter()
        raw_response = backend.generate(prompt)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        print(f"\nResponse received in {elapsed_ms} ms")
        print(f"Response (first 500 chars):\n{raw_response[:500]}...")

        # Parse the model response.
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

        # Save results to output/stage9/.
        serve_result = {
            "run_id": "stage9_serve",
            "model_path": model_path,
            "server_binary": _LLAMA_SERVER_EXE,
            "port": args.port,
            "backend": "llama-server",
            "timestamp": datetime.now(UTC).isoformat(),
            "sample": SAMPLE_REQUEST,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed": parsed,
            "predicted_cwe": result.predicted_cwe if hasattr(result, "predicted_cwe") else None,
            "predicted_severity": (
                result.predicted_severity
                if hasattr(result, "predicted_cwe")
                else None
            ),
            "parse_error": result.reason if not hasattr(result, "predicted_cwe") else None,
            "runtime_ms": elapsed_ms,
            "model_info": backend.model_info,
        }

        output_file = output_dir / "serve_result.json"
        output_file.write_text(json.dumps(serve_result, indent=2, default=str), encoding="utf-8")
        print(f"\nResults saved to {output_file}")

        print("\n=== Stage 9 Serving Summary ===")
        print("  Backend: llama-server (subprocess via HTTP)")
        print(f"  Model:   {model_path}")
        print(f"  Port:    {args.port}")
        print("  Request: real HTTP POST to /completion")
        print(f"  Latency: {elapsed_ms} ms")
        print(f"  Parsed:  {parsed}")
        if parsed:
            print(f"  CWE:     {result.predicted_cwe}")
            print(f"  Severity: {result.predicted_severity}")

    finally:
        backend.close()
        print("\nllama-server subprocess stopped.")

    print("\n=== Stage 9 Complete ===")


if __name__ == "__main__":
    main()
