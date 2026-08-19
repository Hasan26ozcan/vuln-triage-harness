"""Unit tests for Stage 6, Tier 1 — deterministic CWE classification."""

from app.evaluation.tier1_deterministic import (
    DEFAULT_TIER1_RULES,
    DeterministicEvaluator,
    PatternRule,
    classify_deterministic,
)
from app.schemas.vuln import VulnSample


def _make_sample(vuln_code: str, cwe: str = "CWE-89") -> VulnSample:
    return VulnSample(
        id="test_001",
        source="cve_real",
        repo_name="test/repo",
        cwe_id=cwe,
        severity="medium",
        language="python",
        vulnerable_code=vuln_code,
        description="Test vulnerability",
    )


class TestPatternRule:
    def test_frozen_dataclass(self):
        rule = PatternRule(
            cwe="CWE-89",
            pattern=r"test",
            confidence=0.9,
            description="test rule",
        )
        # frozen=True prevents attribute assignment
        try:
            rule.cwe = "CWE-79"
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass

    def test_rule_has_all_fields(self):
        rule = DEFAULT_TIER1_RULES[0]
        assert rule.cwe
        assert rule.pattern
        assert 0.0 <= rule.confidence <= 1.0
        assert rule.description


class TestDeterministicEvaluator:
    def test_default_rules_loaded(self):
        evaluator = DeterministicEvaluator()
        assert len(evaluator.rules) == len(DEFAULT_TIER1_RULES)

    def test_custom_rules_injected(self):
        custom = [
            PatternRule("CWE-89", r"execute", 0.9, "test"),
        ]
        evaluator = DeterministicEvaluator(rules=custom)
        assert len(evaluator.rules) == 1

    def test_sqli_fstring_detected(self):
        code = 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'
        result = classify_deterministic(code, sample_id="t1")
        assert result.predicted_cwe == "CWE-89"
        assert result.confidence > 0.0
        assert "f-string" in result.matched_pattern

    def test_sqli_concat_detected(self):
        code = 'query = "SELECT * FROM u WHERE id=" + str(uid)\ncursor.execute(query)'
        result = classify_deterministic(code, sample_id="t2")
        assert result.predicted_cwe == "CWE-89"

    def test_xss_innerhtml_detected(self):
        code = 'document.getElementById("x").innerHTML = user_input;'
        result = classify_deterministic(code, sample_id="t3")
        assert result.predicted_cwe == "CWE-79"
        assert "innerHTML" in result.matched_pattern

    def test_xss_document_write_detected(self):
        code = "document.write(userInput);"
        result = classify_deterministic(code, sample_id="t4")
        assert result.predicted_cwe == "CWE-79"

    def test_path_traversal_open_detected(self):
        code = 'with open("/safe/" + filename, "r") as f: pass'
        result = classify_deterministic(code, sample_id="t5")
        assert result.predicted_cwe == "CWE-22"

    def test_command_injection_shell_true(self):
        code = "subprocess.run(cmd, shell=True)"
        result = classify_deterministic(code, sample_id="t6")
        assert result.predicted_cwe == "CWE-78"
        assert "shell=True" in result.matched_pattern

    def test_command_injection_os_system(self):
        code = "os.system(user_input)"
        result = classify_deterministic(code, sample_id="t7")
        assert result.predicted_cwe == "CWE-78"

    def test_integer_overflow_bitshift(self):
        code = "return value << scale"
        result = classify_deterministic(code, sample_id="t8")
        assert result.predicted_cwe == "CWE-190"

    def test_deserialization_pickle(self):
        code = "data = pickle.loads(user_data)"
        result = classify_deterministic(code, sample_id="t9")
        assert result.predicted_cwe == "CWE-502"
        assert result.confidence == 0.99

    def test_deserialization_yaml_unsafe(self):
        code = "data = yaml.load(raw_data, Loader=yaml.Loader)"
        result = classify_deterministic(code, sample_id="t10")
        assert result.predicted_cwe == "CWE-502"

    def test_no_match_returns_none_cwe(self):
        code = "print('hello world')"
        result = classify_deterministic(code, sample_id="t11")
        assert result.predicted_cwe is None
        assert result.confidence == 0.0
        assert result.matched_pattern is None
        assert result.num_patterns_matched == 0

    def test_multiple_matches_picks_highest_confidence(self):
        # Both sqli and xss patterns could match
        code = 'cursor.execute(f"SELECT *")\ndocument.write(userInput)'
        result = classify_deterministic(code, sample_id="t12")
        assert result.num_patterns_matched >= 2

    def test_evaluate_single_sample(self):
        evaluator = DeterministicEvaluator()
        sample = _make_sample("pickle.loads(data)")
        result = evaluator.evaluate(sample)
        assert result.sample_id == sample.id
        assert result.predicted_cwe == "CWE-502"

    def test_evaluate_all_batch(self):
        evaluator = DeterministicEvaluator()
        samples = [
            _make_sample("pickle.loads(data)", "CWE-502"),
            _make_sample("os.system(cmd)", "CWE-78"),
            _make_sample("print('safe')", "CWE-89"),
        ]
        results = evaluator.evaluate_all(samples)
        assert len(results) == 3
        assert results[0].predicted_cwe == "CWE-502"
        assert results[1].predicted_cwe == "CWE-78"
        assert results[2].predicted_cwe is None

    def test_all_cwe_classes_covered(self):
        """Every CWE in the gold set should have at least one matching rule."""
        evaluator = DeterministicEvaluator()
        cwes_covered = {rule.cwe for rule in evaluator.rules}
        expected = {"CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"}
        assert cwes_covered == expected

    def test_classify_deterministic_convenience(self):
        code = "obj = pickle.loads(data)"
        result = classify_deterministic(code, sample_id="convenience_1")
        assert result.predicted_cwe == "CWE-502"
        assert result.num_patterns_matched >= 1
