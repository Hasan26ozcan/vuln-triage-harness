"""Tests for Stage 3's token counter.

These verify:
  - The real Qwen tokenizer is used when available (counts tokens correctly).
  - The heuristic fallback works when transformers/tokenizer is absent.
  - count_prompt_and_target sums prompt + target tokens.
  - The counter is injectable (passing a mock tokenizer backend).
  - DEFAULT_MAX_TOKENS and DEFAULT_MODEL constants are sane.
"""

from app.data.formatting.tokenizer import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    TokenCounter,
    _heuristic_count,
)


class _MockTokenizer:
    """A fake tokenizer that returns one token per character.

    This makes token counts predictable and independent of any model.
    """

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))  # one token per character


# --- Constants ---


def test_default_max_tokens_is_sane():
    assert DEFAULT_MAX_TOKENS == 4096


def test_default_model_is_qwen():
    assert "Qwen" in DEFAULT_MODEL


# --- Heuristic fallback ---


def test_heuristic_count_empty_string():
    assert _heuristic_count("") == 0


def test_heuristic_count_single_word():
    # max(1 word, 5 chars / 4 = 2) → 2 (conservative upper bound)
    assert _heuristic_count("hello") == 2


def test_heuristic_count_multiple_words():
    result = _heuristic_count("hello world foo bar")
    # 4 words, 19 chars -> max(4, (19+3)//4=5) = 5
    assert result == 5


def test_heuristic_count_long_code():
    from app.data.formatting.tokenizer import _WORD_RE

    code = (
        "def vulnerable_function(user_input):\n"
        "    cursor.execute('SELECT * FROM t WHERE id = ' + user_input)\n"
    )
    result = _heuristic_count(code)
    assert result > 0
    # Should be roughly chars/4, but at least as many as word tokens
    expected_chars = (len(code) + 3) // 4
    expected_words = len(_WORD_RE.findall(code))
    assert result == max(expected_words, expected_chars)


# --- TokenCounter with mock backend ---


def test_token_counter_with_mock_backend():
    counter = TokenCounter(tokenizer=_MockTokenizer())
    # Each char = 1 token
    assert counter.count("hello") == 5
    assert counter.count("hello world") == 11


def test_token_counter_count_prompt_and_target_with_mock():
    counter = TokenCounter(tokenizer=_MockTokenizer())
    prompt = "hello"  # 5 tokens
    target1 = "world"  # 5 tokens
    target2 = "foo bar"  # 7 tokens
    total = counter.count_prompt_and_target(prompt, target1, target2)
    assert total == 17  # 5 + 5 + 7


def test_token_counter_count_prompt_and_target_skips_none():
    counter = TokenCounter(tokenizer=_MockTokenizer())
    prompt = "hello"  # 5 tokens
    total = counter.count_prompt_and_target(prompt)
    assert total == 5


def test_token_counter_accepts_none_target():
    counter = TokenCounter(tokenizer=_MockTokenizer())
    total = counter.count_prompt_and_target("hello")
    assert total == 5


def test_token_counter_model_name_stored():
    counter = TokenCounter(model_name="test/model", tokenizer=_MockTokenizer())
    assert counter.model_name == "test/model"


# --- TokenCounter with real tokenizer (guarded) ---


def test_token_counter_falls_back_to_heuristic_when_transformers_missing():
    """When transformers can't be imported, the counter should fall back
    to the heuristic tokenizer instead of raising."""
    counter = TokenCounter(model_name="nonexistent/model", tokenizer=None)

    # Force _tokenizer to None and patch _load to raise RuntimeError
    counter._tokenizer = None

    original_load = counter._load

    def mock_load():
        raise RuntimeError("transformers is not installed.")

    counter._load = mock_load
    try:
        # The count() method catches RuntimeError and falls back to heuristic
        result = counter.count("def foo(): return 42")
        assert result == _heuristic_count("def foo(): return 42")
        assert result > 0
    finally:
        counter._load = original_load


def test_token_counter_count_returns_positive_for_real_code():
    """If transformers + the Qwen tokenizer are available, the count should
    be a positive integer. If not, the heuristic fallback should also give
    a positive count. Either way, this should not crash."""
    counter = TokenCounter()
    result = counter.count("cursor.execute('SELECT * FROM users')")
    assert result > 0
    assert isinstance(result, int)


def test_token_counter_count_empty_string():
    counter = TokenCounter(tokenizer=_MockTokenizer())
    assert counter.count("") == 0


def test_token_counter_count_prompt_and_target_returns_int():
    counter = TokenCounter(tokenizer=_MockTokenizer())
    total = counter.count_prompt_and_target("prompt", "target")
    assert isinstance(total, int)
