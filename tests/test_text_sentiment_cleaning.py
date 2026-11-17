import pytest

from ml.text_sentiment import _clean_text


@pytest.mark.parametrize(
    "text",
    ["我今天很开心", "오늘 정말 행복해요", "今日はとても嬉しい"],
)
def test_clean_text_keeps_non_latin_scripts(text):
    assert _clean_text(text) == text


def test_clean_text_keeps_non_latin_words_in_mixed_caption():
    assert _clean_text("日本 trip was wonderful") == "日本 trip was wonderful"


@pytest.mark.parametrize("emoji", ["😀", "☀", "⭐", "✨", "🌄", "Ⓜ", "▶"])
def test_clean_text_still_strips_emoji(emoji):
    assert _clean_text(f"Great {emoji} day #blessed") == "Great day blessed"
