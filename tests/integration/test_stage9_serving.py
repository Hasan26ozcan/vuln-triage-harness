"""Integration tests for Stage 9 — air-gapped serving layer.

Exercises the full serving pipeline end-to-end via the CLI and the FastAPI
TestClient, using ``backend_type="mock"`` so no ML dependencies are required:

* **CLI dry-run** — ``stage9 serve --dry-run`` prints config + warnings.
* **CLI analyze (single)** — ``stage9 serve --analyze -i file.json`` reads
  a single ``ServeRequest`` JSON, runs it through ``MockServingBackend``,
  and writes the ``ServeResponse`` JSON to ``--output-file``.
* **CLI batch** — ``stage9 serve --batch -i file.json`` reads a JSON array,
  serves it, and writes results to ``--output-file``.
* **API single** — ``POST /api/v1/serve`` with a ``ServeRequest``.
* **API batch** — ``POST /api/v1/serve/batch`` with a ``BatchServeRequest``.
* **API manifest** — ``GET /api/v1/manifest`` after serving.
* **End-to-end parse** — the mock returns valid JSON, the server parses
  it, and the response has the expected ``predicted_cwe`` / ``severity``.

No GPU, no real model, no external network — just mock + FastAPI TestClient.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

# Import the shared evaluation CLI (which now includes stage9 subcommand)
eval_cli = pytest.importorskip("app.evaluation.cli")
runner = CliRunner()

MOCK_RESPONSE = (
    '{"cwe_id": "CWE-89", "severity": "high", '
    '"explanation": "SQL injection via string concatenation.", '
    '"patch_diff": ""}'
)


class TestStage9CLIDryRun:
    """CLI dry-run mode — prints config without starting server."""

    def test_dry_run_prints_config(self, tmp_path):
        result = runner.invoke(
            eval_cli.app,
            [
                "stage9",
                "serve",
                "--dry-run",
                "--backend",
                "mock",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Stage 9 Serving Config (dry-run)" in result.output
        assert "Backend type: mock" in result.output
        assert "Run name:" in result.output

    def test_dry_run_llama_cpp_warnings(self, tmp_path):
        """llama.cpp backend with no model_path should show a warning."""
        result = runner.invoke(
            eval_cli.app,
            [
                "stage9",
                "serve",
                "--dry-run",
                "--backend",
                "llama.cpp",
            ],
        )
        assert result.exit_code == 0
        assert "model_path is empty" in result.output


class TestStage9CLIAnalyze:
    """CLI single-sample analysis mode (--analyze)."""

    def test_analyze_single_sample(self, tmp_path):
        """--analyze reads a JSON ServeRequest and writes ServeResponse JSON."""
        req = {
            "sample_id": "cli-test-001",
            "vulnerable_code": "cursor.execute('SELECT * FROM users WHERE id = ' + user_id)",
            "language": "python",
            "description": "SQL injection",
        }
        input_file = tmp_path / "sample.json"
        input_file.write_text(json.dumps(req), encoding="utf-8")
        output_file = tmp_path / "result.json"

        result = runner.invoke(
            eval_cli.app,
            [
                "stage9",
                "serve",
                "--backend",
                "mock",
                "--analyze",
                "--input-file",
                str(input_file),
                "--output-file",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert data["sample_id"] == "cli-test-001"
        assert data["predicted_cwe"] == "CWE-89"
        assert data["predicted_severity"] == "high"
        assert "SQL injection" in data["explanation"]
        assert data["runtime_ms"] is not None

    def test_analyze_no_input_file_fails(self, tmp_path):
        """--analyze without --input-file should exit with error."""
        result = runner.invoke(
            eval_cli.app,
            ["stage9", "serve", "--backend", "mock", "--analyze"],
        )
        assert result.exit_code != 0
        assert "input-file" in result.output.lower() or "required" in result.output.lower()

    def test_analyze_nonexistent_file_fails(self, tmp_path):
        result = runner.invoke(
            eval_cli.app,
            [
                "stage9",
                "serve",
                "--backend",
                "mock",
                "--analyze",
                "--input-file",
                str(tmp_path / "nonexistent.json"),
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_analyze_writes_output_file(self, tmp_path):
        """--output-file should write the ServeResponse JSON to disk."""
        req = {
            "vulnerable_code": "eval(something)",
            "language": "python",
        }
        input_file = tmp_path / "sample.json"
        input_file.write_text(json.dumps(req), encoding="utf-8")
        output_file = tmp_path / "result.json"

        result = runner.invoke(
            eval_cli.app,
            [
                "stage9",
                "serve",
                "--backend",
                "mock",
                "--analyze",
                "--input-file",
                str(input_file),
                "--output-file",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "predicted_cwe" in data

    def test_analyze_sample_without_id_generates_id(self, tmp_path):
        req = {"vulnerable_code": "x = 1\n", "language": "python"}
        input_file = tmp_path / "sample.json"
        input_file.write_text(json.dumps(req), encoding="utf-8")
        output_file = tmp_path / "result.json"

        result = runner.invoke(
            eval_cli.app,
            [
                "stage9",
                "serve",
                "--backend",
                "mock",
                "--analyze",
                "--input-file",
                str(input_file),
                "--output-file",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["sample_id"] is not None
        assert "sample-" in data["sample_id"]


class TestStage9CLIBatch:
    """CLI batch mode (--batch)."""

    def test_batch_multiple_samples(self, tmp_path):
        """--batch reads a JSON array and serves all samples."""
        requests = [
            {
                "vulnerable_code": "cursor.execute('SELECT * FROM u WHERE id = ' + uid)",
                "language": "python",
            },
            {"vulnerable_code": "eval(input())", "language": "python"},
        ]
        input_file = tmp_path / "batch.json"
        input_file.write_text(json.dumps(requests), encoding="utf-8")
        output_file = tmp_path / "batch_result.json"

        result = runner.invoke(
            eval_cli.app,
            [
                "stage9",
                "serve",
                "--backend",
                "mock",
                "--batch",
                "--input-file",
                str(input_file),
                "--output-file",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert len(data["responses"]) == 2
        assert data["manifest"]["num_requests"] == 2
        assert data["manifest"]["backend_type"] == "mock"

    def test_batch_writes_output_file(self, tmp_path):
        """--batch --output-file writes results to disk."""
        requests = [
            {"vulnerable_code": "eval(x)", "language": "python"},
        ]
        input_file = tmp_path / "batch.json"
        input_file.write_text(json.dumps(requests), encoding="utf-8")
        output_file = tmp_path / "batch_result.json"

        result = runner.invoke(
            eval_cli.app,
            [
                "stage9",
                "serve",
                "--backend",
                "mock",
                "--batch",
                "--input-file",
                str(input_file),
                "--output-file",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert len(data["responses"]) == 1


# ---------------------------------------------------------------------------
# API integration tests (via FastAPI TestClient)
# ---------------------------------------------------------------------------

TestClient = pytest.importorskip("fastapi.testclient").TestClient
from app.schemas.serving import BatchServeRequest, ServeRequest  # noqa: E402
from app.serving.api import create_app  # noqa: E402
from app.serving.config import ServingConfig  # noqa: E402


@pytest.fixture
def api_client():
    """Create a TestClient with mock backend for API integration tests."""
    config = ServingConfig(backend_type="mock")
    test_app = create_app(config)
    return TestClient(test_app)


class TestStage9API:
    """End-to-end tests through the FastAPI TestClient."""

    def test_healthz(self, api_client):
        resp = api_client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_manifest_endpoint(self, api_client):
        resp = api_client.get("/api/v1/manifest")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert "backend_type" in data

    def test_serve_single_via_api(self, api_client):
        req = ServeRequest(
            sample_id="api-integration-001",
            vulnerable_code="cursor.execute('SELECT * FROM users WHERE id = ' + user_id)",
            language="python",
            description="SQL injection",
        )
        resp = api_client.post("/api/v1/serve", json=req.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert data["sample_id"] == "api-integration-001"
        assert data["run_id"] is not None
        assert data["predicted_cwe"] == "CWE-89"
        assert data["predicted_severity"] == "high"
        assert data["runtime_ms"] is not None

    def test_serve_batch_via_api(self, api_client):
        batch = BatchServeRequest(
            requests=[
                ServeRequest(
                    sample_id="batch-001",
                    vulnerable_code="cursor.execute('SELECT * FROM u WHERE id = ' + user_id)",
                    language="python",
                ),
                ServeRequest(
                    sample_id="batch-002",
                    vulnerable_code="subprocess.call(user_input, shell=True)",
                    language="python",
                ),
            ]
        )
        resp = api_client.post("/api/v1/serve/batch", json=batch.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["responses"]) == 2
        assert data["responses"][0]["sample_id"] == "batch-001"
        assert data["responses"][1]["sample_id"] == "batch-002"
        assert data["manifest"]["num_requests"] == 2
        assert data["manifest"]["backend_type"] == "mock"
        assert data["responses"][0]["run_id"] == data["responses"][1]["run_id"]

    def test_serve_validation_error(self, api_client):
        """Missing vulnerable_code should return 422."""
        resp = api_client.post("/api/v1/serve", json={"language": "python"})
        assert resp.status_code == 422

    def test_batch_empty(self, api_client):
        batch = BatchServeRequest(requests=[])
        resp = api_client.post("/api/v1/serve/batch", json=batch.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert data["responses"] == []
        assert data["manifest"]["num_requests"] == 0


# ---------------------------------------------------------------------------
# End-to-end: CLI → MockServingBackend → parse_prediction → ServeResponse
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Verify the full pipeline works from prompt to parsed response."""

    def test_mock_backend_returns_expected_cwe(self):
        """The MockServingBackend returns CWE-89, and the server correctly
        parses it through parse_prediction (Stage 4) into a ServeResponse."""
        from app.serving.backends import MockServingBackend
        from app.serving.config import ServingConfig
        from app.serving.serve import VulnerabilityServer

        backend = MockServingBackend()
        config = ServingConfig(backend_type="mock")
        server = VulnerabilityServer(backend=backend, config=config)

        req = ServeRequest(
            vulnerable_code="cursor.execute('SELECT * FROM users WHERE id = ' + user_id)",
            language="python",
            cwe_id="CWE-89",
            severity="high",
        )
        resp = server.serve_sample(req)

        assert resp.predicted_cwe == "CWE-89"
        assert resp.predicted_severity == "high"
        assert "SQL injection" in resp.explanation
        assert resp.patch_diff == ""
        assert resp.run_id == server.run_id

    def test_server_backend_model_info_in_manifest(self):
        """The manifest should reflect MockServingBackend's model_info."""
        from app.serving.backends import MockServingBackend
        from app.serving.config import ServingConfig
        from app.serving.serve import VulnerabilityServer

        backend = MockServingBackend()
        config = ServingConfig(backend_type="mock")
        server = VulnerabilityServer(backend=backend, config=config)

        req = ServeRequest(vulnerable_code="x = 1\n")
        server.serve_sample(req)

        manifest = server.get_manifest()
        assert manifest["backend_type"] == "mock"
        assert manifest["model_path"] == "mock"
        assert manifest["num_requests"] == 1

    def test_cli_and_api_produce_same_result(self, tmp_path):
        """Same input via CLI and API should produce equivalent CWE predictions."""
        req = {
            "sample_id": "consistency-test",
            "vulnerable_code": "eval(user_input)",
            "language": "python",
        }
        input_file = tmp_path / "sample.json"
        input_file.write_text(json.dumps(req), encoding="utf-8")
        cli_output = tmp_path / "cli_result.json"

        # Via CLI
        cli_result = runner.invoke(
            eval_cli.app,
            [
                "stage9",
                "serve",
                "--backend",
                "mock",
                "--analyze",
                "--input-file",
                str(input_file),
                "--output-file",
                str(cli_output),
            ],
        )
        assert cli_result.exit_code == 0, cli_result.output
        cli_data = json.loads(cli_output.read_text())

        # Via API
        config = ServingConfig(backend_type="mock")
        test_app = create_app(config)
        client = TestClient(test_app)
        api_resp = client.post("/api/v1/serve", json=req)
        assert api_resp.status_code == 200
        api_data = api_resp.json()

        # Both should return the same mock prediction
        assert cli_data["predicted_cwe"] == api_data["predicted_cwe"]
        assert cli_data["predicted_severity"] == api_data["predicted_severity"]
