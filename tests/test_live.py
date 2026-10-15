import pytest
from dotenv import load_dotenv

from app.graph import build_graph
from app.llm import make_llm


@pytest.mark.live
def test_live_end_to_end():
    load_dotenv()
    result = build_graph(make_llm()).invoke({"prompt": "What is 2 + 2?"})
    assert result["answer"].strip()
    assert 0 <= result["confidence"] <= 100
    assert not result["review_reasoning"].startswith("review failed"), result["review_reasoning"]
    assert result["outcome"] in {"success", "error"}
