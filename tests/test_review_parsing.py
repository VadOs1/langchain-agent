import pytest
from langchain_core.messages import AIMessage

from app.graph import ReviewResult, message_text, parse_review


def test_message_text_plain_string():
    assert message_text(AIMessage(content="hello")) == "hello"


def test_message_text_drops_reasoning_blocks():
    msg = AIMessage(
        content=[
            {"type": "reasoning_content", "reasoning_content": {"text": "thinking..."}},
            {"type": "text", "text": "Paris"},
            {"type": "text", "text": " is the capital."},
        ]
    )
    assert message_text(msg) == "Paris is the capital."


def test_parse_bare_json():
    assert parse_review('{"confidence": 92, "reasoning": "correct"}') == ReviewResult(
        confidence=92, reasoning="correct"
    )


def test_parse_json_in_code_fence_with_prose():
    text = 'Here is my review:\n```json\n{"confidence": 40, "reasoning": "vague"}\n```\nThanks.'
    assert parse_review(text).confidence == 40


def test_parse_skips_non_json_braces():
    text = 'Note {not json} then {"confidence": 70, "reasoning": "ok"}'
    assert parse_review(text).confidence == 70


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"confidence": 150, "reasoning": "too high"}',
        '{"confidence": -1, "reasoning": "negative"}',
        '{"confidence": "92%", "reasoning": "string"}',
        '{"reasoning": "missing confidence"}',
        "",
    ],
)
def test_parse_rejects_invalid(text):
    with pytest.raises(ValueError):
        parse_review(text)
