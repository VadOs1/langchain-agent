"""LangGraph pipeline: generate an answer, review it, route on confidence."""

import json
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

CONFIDENCE_THRESHOLD = 85

GENERATE_SYSTEM = "Answer the user's question accurately and concisely."

REVIEW_SYSTEM = (
    "You are a strict reviewer. Given a question and a proposed answer, judge how "
    "confident you are that the answer is correct, complete, and directly addresses "
    "the question. Reply with only a JSON object and no other text, in exactly this "
    'form: {"confidence": <integer 0-100>, "reasoning": "<one or two sentences>"}'
)


class GraphState(TypedDict, total=False):
    prompt: str
    answer: str
    confidence: int
    review_reasoning: str
    outcome: Literal["success", "error"]


def route_after_review(state: GraphState) -> Literal["success", "error"]:
    """Send the run to success only when confidence is strictly above the threshold."""
    return "success" if state.get("confidence", 0) > CONFIDENCE_THRESHOLD else "error"


class ReviewResult(BaseModel):
    confidence: int = Field(ge=0, le=100)
    reasoning: str


def message_text(message: BaseMessage) -> str:
    """Return only the text of a chat reply, dropping reasoning and other blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


_DECODER = json.JSONDecoder()


def parse_review(text: str) -> ReviewResult:
    """Parse the first JSON object in the reviewer's reply.

    Tolerates surrounding prose and code fences. Raises ValueError (pydantic's
    ValidationError is a subclass) when no valid ReviewResult is found.
    """
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return ReviewResult.model_validate(obj)
    raise ValueError("no JSON object found in review reply")


def _print_result(label: str, state: GraphState) -> None:
    print(f"=== {label} (confidence {state['confidence']}) ===")
    print(f"Prompt:  {state['prompt']}")
    print(f"Answer:  {state['answer']}")
    print(f"Review:  {state['review_reasoning']}")


def build_graph(llm: BaseChatModel):
    """Build the generate → review → success|error graph around the given chat model."""

    def generate(state: GraphState) -> GraphState:
        reply = llm.invoke(
            [SystemMessage(content=GENERATE_SYSTEM), HumanMessage(content=state["prompt"])]
        )
        return {"answer": message_text(reply)}

    def review(state: GraphState) -> GraphState:
        question = f"Question:\n{state['prompt']}\n\nProposed answer:\n{state['answer']}"
        reply = llm.invoke([SystemMessage(content=REVIEW_SYSTEM), HumanMessage(content=question)])
        try:
            result = parse_review(message_text(reply))
        except ValueError as exc:
            reason = str(exc).splitlines()[0]
            return {"confidence": 0, "review_reasoning": f"review failed: {reason}"}
        return {"confidence": result.confidence, "review_reasoning": result.reasoning}

    def success(state: GraphState) -> GraphState:
        _print_result("SUCCESS", state)
        return {"outcome": "success"}

    def error(state: GraphState) -> GraphState:
        _print_result("ERROR", state)
        return {"outcome": "error"}

    graph = StateGraph(GraphState)
    graph.add_node("generate", generate)
    graph.add_node("review", review)
    graph.add_node("success", success)
    graph.add_node("error", error)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "review")
    graph.add_conditional_edges("review", route_after_review, {"success": "success", "error": "error"})
    graph.add_edge("success", END)
    graph.add_edge("error", END)
    return graph.compile()
