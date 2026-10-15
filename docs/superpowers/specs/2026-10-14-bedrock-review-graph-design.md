# Bedrock Review Graph — Design

**Date:** 2026-10-14
**Status:** Approved in conversation; pending written-spec review

## Purpose

A learning/demo project to explore LangGraph with Claude on AWS Bedrock. A CLI
that takes a prompt, generates an answer, has the model review its own answer
with a confidence score, and routes to a success or error node based on that
score. No API server, persistence, or production hardening.

## Requirements

- Python, LangGraph (`StateGraph`), `langchain-aws` (`ChatBedrockConverse`).
- Model: Claude Sonnet 5.5 on AWS Bedrock, used for both generation and review.
- Four nodes: `generate`, `review`, `success`, `error`.
- Route to `success` when confidence **> 85** (strictly greater); otherwise `error`.
- Single pass — no retry loop from `error` back to `generate`.
- Run as `python main.py "your prompt"`.

## Architecture

```
START → generate → review ──(confidence > 85)──→ success → END
                          └─(confidence ≤ 85)──→ error   → END
```

### State

```python
class GraphState(TypedDict, total=False):
    prompt: str
    answer: str
    confidence: int          # 0–100
    review_reasoning: str
    outcome: Literal["success", "error"]
```

### Nodes

| Node | Input | Output | Behavior |
|---|---|---|---|
| `generate` | `prompt` | `answer` | One Sonnet 5.5 call answering the prompt. |
| `review` | `prompt`, `answer` | `confidence`, `review_reasoning` | Separate Sonnet 5.5 call acting as a judge; returns `ReviewResult` (Pydantic: `confidence: int` constrained 0–100, `reasoning: str`). |
| `success` | full state | `outcome="success"` | Prints formatted result. Placeholder for future real actions. |
| `error` | full state | `outcome="error"` | Prints formatted result including review reasoning. |

### Router

`route_after_review(state) -> Literal["success", "error"]` — pure function,
returns `"success"` iff `state["confidence"] > CONFIDENCE_THRESHOLD` where
`CONFIDENCE_THRESHOLD = 85` is a module-level constant.

### Dependency injection

`build_graph(llm: BaseChatModel) -> CompiledGraph`. Nodes close over the passed
model; they never construct a client. `main.py` passes the real Bedrock client;
tests pass a fake.

## File layout

```
pyproject.toml          # deps: langgraph, langchain-aws, python-dotenv, pydantic; dev: pytest
.env.example
main.py                 # CLI entrypoint
src/app/__init__.py
src/app/llm.py          # make_llm(): ChatBedrockConverse from env config
src/app/graph.py        # GraphState, ReviewResult, nodes, router, build_graph
tests/test_router.py
tests/test_graph.py     # fake-model end-to-end tests
tests/test_live.py      # @pytest.mark.live, skipped by default
```

## Configuration

Loaded from `.env` via `python-dotenv`:

- `AWS_REGION` — required.
- `AWS_PROFILE` — optional; otherwise the standard boto3 credential chain.
- `BEDROCK_MODEL_ID` — defaults to the Sonnet 5.5 Bedrock ID. Depending on
  account/region this is either `anthropic.claude-sonnet-5-5` or a cross-region
  inference profile such as `us.anthropic.claude-sonnet-5-5`. The working default
  is confirmed with one live call during implementation and documented in
  `.env.example`.

## Sonnet 5.5 constraints

1. **Forced `tool_choice` (`any` / named tool) returns HTTP 400.** LangChain's
   default `with_structured_output` forces a tool call, so it is not used. The
   review node instead instructs the model to reply with only a JSON object
   matching `ReviewResult`, extracts the first `{...}` block from the reply
   (tolerating surrounding text or code fences), and validates it with
   `ReviewResult.model_validate_json`. This avoids `tool_choice` entirely and
   works identically with the fake model in tests.
2. **Non-default `temperature` returns HTTP 400.** Do not set `temperature`.
3. Adaptive thinking is on by default; no thinking config is passed.

## Error handling

- **Review output fails to parse or validate** → `review` sets `confidence=0`
  and `review_reasoning="review failed: <short error>"`. The run routes to
  `error`; a malformed review never counts as a pass.
- **AWS/Bedrock exceptions** (missing credentials, model access denied,
  throttling) are not caught inside the graph. `main.py` catches them at top
  level and prints a one-line hint (check `AWS_PROFILE`, `AWS_REGION`, and
  Bedrock model access), exiting non-zero.
- **No prompt argument** → usage message, exit code 2.

## Testing

- `test_router.py`: confidence 85 → `error`, 86 → `success`, 0 and 100 edges.
- `test_graph.py`: using LangChain's `GenericFakeChatModel` with scripted
  responses —
  - high-confidence review → `outcome == "success"`;
  - low-confidence review → `outcome == "error"`;
  - malformed review JSON → `outcome == "error"`, `confidence == 0`.
- `test_live.py`: one end-to-end call against Bedrock, marked `live` and skipped
  unless run with `pytest -m live`.

## Out of scope

Retry/regenerate loop, HTTP API, persistence, streaming output, observability
(LangSmith), multiple models.
