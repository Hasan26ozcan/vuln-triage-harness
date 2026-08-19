"""Unit tests for Stage 9 — serving schemas (app/schemas/serving.py).

Covers all Pydantic models with happy paths, required-field validation,
and edge cases so that ``schemas/serving.py`` has 100 % line coverage.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.serving import (
    BatchServeRequest,
    BatchServeResponse,
    ServeManifest,
    ServeRequest,
    ServeResponse,
)
from app.schemas.vuln import StaticFinding

# --- ServeRequest ---


class TestServeRequest:
    def test_valid_minimal(self):
        req = ServeRequest(vulnerable_code="x = 1\n")
        assert req.vulnerable_code == "x = 1\n"
        assert req.language == "python"
        assert req.sample_id is None
        assert req.cwe_id is None
        assert req.severity is None
        assert req.description == ""
        assert req.static_findings == []
        assert req.temperature is None
        assert req.max_new_tokens is None

    def test_valid_with_optional_fields(self):
        req = ServeRequest(
            sample_id="sample-001",
            vulnerable_code="eval(user_input)",
            language="python",
            cwe_id="CWE-95",
            severity="high",
            description="Eval injection",
            static_findings=[
                StaticFinding(
                    tool="semgrep",
                    rule_id="python.lang.security.eval",
                    message="eval call",
                    line_range=(5, 5),
                )
            ],
            temperature=0.1,
            max_new_tokens=512,
        )
        assert req.sample_id == "sample-001"
        assert req.cwe_id == "CWE-95"
        assert len(req.static_findings) == 1

    def test_required_vulnerable_code(self):
        with pytest.raises(ValidationError):
            ServeRequest()


# --- ServeResponse ---


class TestServeResponse:
    def test_valid_minimal(self):
        resp = ServeResponse(
            sample_id="s1",
            run_id="r1",
            predicted_cwe="CWE-89",
            predicted_severity="high",
            explanation="SQL injection.",
            patch_diff="",
        )
        assert resp.sample_id == "s1"
        assert resp.run_id == "r1"
        assert resp.predicted_cwe == "CWE-89"
        assert resp.predicted_severity == "high"
        assert resp.explanation == "SQL injection."
        assert resp.patch_diff == ""
        assert resp.runtime_ms is None

    def test_valid_with_runtime(self):
        resp = ServeResponse(
            sample_id="s1",
            run_id="r1",
            predicted_cwe="CWE-79",
            predicted_severity="medium",
            explanation="XSS",
            patch_diff="---",
            runtime_ms=42.5,
        )
        assert resp.runtime_ms == 42.5

    def test_json_serialization(self):
        resp = ServeResponse(
            sample_id="s1",
            run_id="r1",
            predicted_cwe="CWE-89",
            predicted_severity="high",
            explanation="SQLi",
            patch_diff="",
            runtime_ms=10.0,
        )
        data = resp.model_dump()
        assert data["sample_id"] == "s1"
        assert data["predicted_cwe"] == "CWE-89"
        json_str = resp.model_dump_json()
        assert "CWE-89" in json_str

    def test_parse_error_response(self):
        resp = ServeResponse(
            sample_id="s1",
            run_id="r1",
            predicted_cwe="PARSE_ERROR",
            predicted_severity="unknown",
            explanation="No JSON object found in response",
            patch_diff="",
        )
        assert resp.predicted_cwe == "PARSE_ERROR"


# --- BatchServeRequest ---


class TestBatchServeRequest:
    def test_empty_batch(self):
        batch = BatchServeRequest()
        assert batch.requests == []

    def test_with_requests(self):
        reqs = [
            ServeRequest(vulnerable_code="a"),
            ServeRequest(vulnerable_code="b"),
        ]
        batch = BatchServeRequest(requests=reqs)
        assert len(batch.requests) == 2

    def test_serialization(self):
        batch = BatchServeRequest(requests=[ServeRequest(vulnerable_code="x")])
        data = batch.model_dump()
        assert len(data["requests"]) == 1


# --- BatchServeResponse ---


class TestBatchServeResponse:
    def test_empty(self):
        resp = BatchServeResponse()
        assert resp.responses == []
        assert resp.manifest == {}

    def test_with_responses_and_manifest(self):
        sr = ServeResponse(
            sample_id="s1",
            run_id="r1",
            predicted_cwe="CWE-89",
            predicted_severity="high",
            explanation="SQLi",
            patch_diff="---",
        )
        resp = BatchServeResponse(
            responses=[sr],
            manifest={"run_id": "r1", "backend_type": "mock", "avg_runtime_ms": 10.0},
        )
        assert len(resp.responses) == 1
        assert resp.manifest["backend_type"] == "mock"


# --- ServeManifest ---


class TestServeManifest:
    def test_valid(self):
        manifest = ServeManifest(
            run_id="run-123",
            backend_type="llama.cpp",
            model_path="/models/qwen.gguf",
            num_requests=5,
            started_at="2024-01-01T00:00:00Z",
            avg_runtime_ms=42.5,
        )
        assert manifest.run_id == "run-123"
        assert manifest.backend_type == "llama.cpp"
        assert manifest.model_path == "/models/qwen.gguf"
        assert manifest.num_requests == 5
        assert manifest.avg_runtime_ms == 42.5

    def test_defaults(self):
        manifest = ServeManifest(
            run_id="r1",
            backend_type="mock",
            model_path="mock",
        )
        assert manifest.num_requests == 0
        assert manifest.started_at == ""
        assert manifest.avg_runtime_ms is None

    def test_serialization(self):
        manifest = ServeManifest(
            run_id="r1",
            backend_type="mock",
            model_path="mock",
            num_requests=3,
            started_at="2024-01-01T00:00:00Z",
            avg_runtime_ms=5.0,
        )
        data = manifest.model_dump()
        assert data["run_id"] == "r1"
        assert data["num_requests"] == 3
        json_str = manifest.model_dump_json()
        assert "avg_runtime_ms" in json_str
