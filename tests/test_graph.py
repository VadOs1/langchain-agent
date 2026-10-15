from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.graph import build_graph


def fake_llm(*replies):
    """Fake chat model returning replies in order: generate first, then review."""
    return GenericFakeChatModel(messages=iter(replies))


def test_high_confidence_routes_to_success():
    llm = fake_llm("Paris.", '{"confidence": 95, "reasoning": "Correct and direct."}')
    result = build_graph(llm).invoke({"prompt": "Capital of France?"})
    assert result["answer"] == "Paris."
    assert result["confidence"] == 95
    assert result["review_reasoning"] == "Correct and direct."
    assert result["outcome"] == "success"


def test_low_confidence_routes_to_error():
    llm = fake_llm("Maybe Lyon?", '{"confidence": 20, "reasoning": "Wrong city."}')
    result = build_graph(llm).invoke({"prompt": "Capital of France?"})
    assert result["confidence"] == 20
    assert result["outcome"] == "error"


def test_threshold_boundary_85_is_error():
    llm = fake_llm("Paris.", '{"confidence": 85, "reasoning": "Borderline."}')
    assert build_graph(llm).invoke({"prompt": "q"})["outcome"] == "error"


def test_malformed_review_routes_to_error():
    llm = fake_llm("Paris.", "I think it's pretty good!")
    result = build_graph(llm).invoke({"prompt": "Capital of France?"})
    assert result["confidence"] == 0
    assert result["review_reasoning"].startswith("review failed: ")
    assert result["outcome"] == "error"


def test_out_of_range_confidence_routes_to_error():
    llm = fake_llm("Paris.", '{"confidence": 150, "reasoning": "Very sure."}')
    result = build_graph(llm).invoke({"prompt": "q"})
    assert result["confidence"] == 0
    assert result["outcome"] == "error"


def test_list_content_with_reasoning_blocks():
    answer = AIMessage(
        content=[
            {"type": "reasoning_content", "reasoning_content": {"text": "hmm"}},
            {"type": "text", "text": "Paris."},
        ]
    )
    review = AIMessage(
        content=[
            {"type": "reasoning_content", "reasoning_content": {"text": "{not json}"}},
            {"type": "text", "text": '{"confidence": 90, "reasoning": "Right."}'},
        ]
    )
    result = build_graph(fake_llm(answer, review)).invoke({"prompt": "q"})
    assert result["answer"] == "Paris."
    assert result["outcome"] == "success"


def test_nodes_print_result(capsys):
    llm = fake_llm("Paris.", '{"confidence": 95, "reasoning": "Correct."}')
    build_graph(llm).invoke({"prompt": "Capital of France?"})
    out = capsys.readouterr().out
    assert "SUCCESS" in out
    assert "Paris." in out
    assert "95" in out
