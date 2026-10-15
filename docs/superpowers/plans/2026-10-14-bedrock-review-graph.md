# Bedrock Review Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI that answers a prompt with Claude Sonnet 5.5 on Bedrock, has the model review its own answer with a 0–100 confidence score, and routes to a success node (> 85) or an error node (≤ 85).

**Architecture:** A four-node LangGraph `StateGraph` (`generate → review → success | error`) built by `build_graph(llm)`, which takes the chat model as a parameter so tests inject `GenericFakeChatModel`. The review node asks for a bare JSON object and validates it with Pydantic — no `with_structured_output`, because Sonnet 5.5 rejects forced `tool_choice`. `main.py` wires the real `ChatBedrockConverse` client from `.env`.

**Tech Stack:** Python ≥ 3.11, uv, langgraph, langchain-aws (`ChatBedrockConverse`), langchain-core, pydantic v2, python-dotenv, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-bedrock-review-graph-design.md`

## Global Constraints

- Model: Claude Sonnet 5.5 on AWS Bedrock, same model for generate and review.
- Route to `success` iff confidence **> 85** (strictly greater); `CONFIDENCE_THRESHOLD = 85` module constant.
- Single pass — no retry loop.
- Never set `temperature` (non-default values return HTTP 400 on Sonnet 5.5).
- Never use `with_structured_output` / forced `tool_choice` (HTTP 400 on Sonnet 5.5).
- Nodes never construct a model client; `build_graph(llm)` receives it.
- Malformed review → `confidence=0`, `review_reasoning` starts with `"review failed: "`, routes to `error`.
- Default test run makes no AWS calls; live test runs only with `pytest -m live`.
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Bedrock returns list content** (adaptive thinking yields reasoning blocks before the text block) → only the `text` blocks become `answer` / get parsed. Pinned in Task 2 (`message_text`) and Task 3 (graph test with list content).
2. **Review JSON wrapped in prose or ```json fences** → still parsed. Pinned in Task 2.
3. **Confidence out of range or wrong type** (e.g. `150`, `"92%"`) → treated as review failure → `error`, never success. Pinned in Task 2 and Task 3.
4. **Empty or whitespace-only prompt** (`python main.py ""`) → usage message, exit 2, no AWS call. Pinned in Task 4.
5. **Missing `AWS_REGION` / bad credentials** → one-line hint on stderr, exit 1, no traceback. Pinned in Task 4.

## File Structure

```
pyproject.toml          # project + deps + pytest config (Task 1)
.env.example            # config template (Task 4, confirmed in Task 5)
main.py                 # CLI: arg handling, top-level error handling (Task 4)
src/app/__init__.py     # empty package marker (Task 1)
src/app/graph.py        # state, router, review parsing, nodes, build_graph (Tasks 1–3)
src/app/llm.py          # make_llm() + ConfigError (Task 4)
tests/test_router.py    # Task 1
tests/test_review_parsing.py  # Task 2
tests/test_graph.py     # Task 3
tests/test_cli.py       # Task 4
tests/test_live.py      # Task 5
README.md               # usage (Task 5)
```

---

### Task 1: Project scaffold, state, and router

**Files:**
- Create: `pyproject.toml`, `src/app/__init__.py`, `src/app/graph.py`, `tests/test_router.py`
- Modify: `.gitignore` (create)

**Interfaces:**
- Produces: `app.graph.CONFIDENCE_THRESHOLD: int = 85`; `app.graph.GraphState` (TypedDict, `total=False`: `prompt: str`, `answer: str`, `confidence: int`, `review_reasoning: str`, `outcome: Literal["success","error"]`); `app.graph.route_after_review(state: GraphState) -> Literal["success","error"]`.

- [ ] **Step 1: Confirm uv is available**

Run: `uv --version`
Expected: prints a version. If missing: `brew install uv`.

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "cakesist-langchain-agent"
version = "0.1.0"
description = "LangGraph demo: generate, self-review, and route on confidence with Claude on Bedrock"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=1.0",
    "langchain-core>=1.0",
    "langchain-aws>=1.0",
    "pydantic>=2.7",
    "python-dotenv>=1.0",
]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/app"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-m 'not live'"
markers = ["live: calls real AWS Bedrock (run with: pytest -m live)"]
```

- [ ] **Step 3: Create `.gitignore` and empty `src/app/__init__.py`**

`.gitignore`:
```
.venv/
__pycache__/
.pytest_cache/
.env
```

`src/app/__init__.py`: empty file.

- [ ] **Step 4: Install**

Run: `uv sync`
Expected: creates `.venv` and `uv.lock`, installs deps and the `app` package.

- [ ] **Step 5: Write the failing router test** — `tests/test_router.py`

```python
import pytest

from app.graph import CONFIDENCE_THRESHOLD, route_after_review


def test_threshold_is_85():
    assert CONFIDENCE_THRESHOLD == 85


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0, "error"), (85, "error"), (86, "success"), (100, "success")],
)
def test_route_after_review(confidence, expected):
    assert route_after_review({"confidence": confidence}) == expected


def test_missing_confidence_routes_to_error():
    assert route_after_review({}) == "error"
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest tests/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.graph'`.

- [ ] **Step 7: Implement** — `src/app/graph.py`

```python
"""LangGraph pipeline: generate an answer, review it, route on confidence."""

from typing import Literal, TypedDict

CONFIDENCE_THRESHOLD = 85


class GraphState(TypedDict, total=False):
    prompt: str
    answer: str
    confidence: int
    review_reasoning: str
    outcome: Literal["success", "error"]


def route_after_review(state: GraphState) -> Literal["success", "error"]:
    """Send the run to success only when confidence is strictly above the threshold."""
    return "success" if state.get("confidence", 0) > CONFIDENCE_THRESHOLD else "error"
```

- [ ] **Step 8: Run to verify it passes**

Run: `uv run pytest tests/test_router.py -v`
Expected: 6 passed.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/app/__init__.py src/app/graph.py tests/test_router.py
git commit -m "feat: scaffold project with graph state and confidence router" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Message text extraction and review parsing

**Files:**
- Modify: `src/app/graph.py`
- Create: `tests/test_review_parsing.py`

**Interfaces:**
- Consumes: `app.graph` module from Task 1.
- Produces: `app.graph.ReviewResult` (Pydantic: `confidence: int` with `0 ≤ x ≤ 100`, `reasoning: str`); `app.graph.message_text(message: BaseMessage) -> str` (joins `str` content or `type == "text"` blocks, drops reasoning/other blocks); `app.graph.parse_review(text: str) -> ReviewResult` (raises `ValueError` — including pydantic `ValidationError` — when no valid object is found).

- [ ] **Step 1: Write the failing tests** — `tests/test_review_parsing.py`

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_review_parsing.py -v`
Expected: FAIL — `ImportError: cannot import name 'ReviewResult'`.

- [ ] **Step 3: Implement** — add to `src/app/graph.py`

Replace the imports at the top with:

```python
import json
from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
```

Append after `route_after_review`:

```python
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
```

Note: `"92%"` fails because pydantic's lax mode only coerces numeric strings like `"92"`; that is intended.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -v`
Expected: all tests in `test_router.py` and `test_review_parsing.py` pass.

- [ ] **Step 5: Commit**

```bash
git add src/app/graph.py tests/test_review_parsing.py
git commit -m "feat: parse reviewer JSON and extract text from chat replies" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Nodes and graph assembly

**Files:**
- Modify: `src/app/graph.py`
- Create: `tests/test_graph.py`

**Interfaces:**
- Consumes: `GraphState`, `route_after_review`, `message_text`, `parse_review` from Tasks 1–2.
- Produces: `app.graph.build_graph(llm: BaseChatModel)` → compiled LangGraph; `.invoke({"prompt": str})` returns a dict with `prompt`, `answer`, `confidence`, `review_reasoning`, `outcome`. The model is called exactly twice per run: generate first, review second.

- [ ] **Step 1: Write the failing tests** — `tests/test_graph.py`

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_graph'`.

- [ ] **Step 3: Implement** — add to `src/app/graph.py`

Replace the imports at the top with:

```python
import json
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
```

Add below `CONFIDENCE_THRESHOLD`:

```python
GENERATE_SYSTEM = "Answer the user's question accurately and concisely."

REVIEW_SYSTEM = (
    "You are a strict reviewer. Given a question and a proposed answer, judge how "
    "confident you are that the answer is correct, complete, and directly addresses "
    "the question. Reply with only a JSON object and no other text, in exactly this "
    'form: {"confidence": <integer 0-100>, "reasoning": "<one or two sentences>"}'
)
```

Append at the end of the file:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -v`
Expected: all tests pass (router, parsing, graph).

- [ ] **Step 5: Commit**

```bash
git add src/app/graph.py tests/test_graph.py
git commit -m "feat: add generate/review/success/error nodes and build_graph" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Bedrock client factory and CLI

**Files:**
- Create: `src/app/llm.py`, `main.py`, `.env.example`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `app.graph.build_graph` from Task 3.
- Produces: `app.llm.ConfigError(RuntimeError)`; `app.llm.DEFAULT_MODEL_ID: str`; `app.llm.make_llm() -> ChatBedrockConverse` (reads `AWS_REGION` required, `AWS_PROFILE` optional, `BEDROCK_MODEL_ID` optional); `main.main(argv: list[str]) -> int` (0 = run completed with either outcome, 1 = config/AWS failure, 2 = usage error).

- [ ] **Step 1: Write the failing tests** — `tests/test_cli.py`

```python
from botocore.exceptions import NoCredentialsError
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

import main
from app.llm import DEFAULT_MODEL_ID, ConfigError, make_llm


def test_make_llm_requires_region(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    try:
        make_llm()
    except ConfigError as exc:
        assert "AWS_REGION" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_make_llm_uses_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("BEDROCK_MODEL_ID", "custom.model-id")
    llm = make_llm()
    assert llm.model_id == "custom.model-id"
    assert llm.region_name == "us-east-1"
    assert llm.temperature is None


def test_make_llm_defaults_model(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("BEDROCK_MODEL_ID", "")
    assert make_llm().model_id == DEFAULT_MODEL_ID


def test_cli_empty_prompt_is_usage_error(capsys, monkeypatch):
    def boom():
        raise AssertionError("must not build a client for an empty prompt")

    monkeypatch.setattr(main, "make_llm", boom)
    assert main.main([]) == 2
    assert main.main(["   "]) == 2
    assert "usage" in capsys.readouterr().err


def test_cli_runs_graph(capsys, monkeypatch):
    llm = GenericFakeChatModel(
        messages=iter(["Paris.", '{"confidence": 95, "reasoning": "Correct."}'])
    )
    monkeypatch.setattr(main, "make_llm", lambda: llm)
    assert main.main(["Capital", "of", "France?"]) == 0
    out = capsys.readouterr().out
    assert "SUCCESS" in out
    assert "Capital of France?" in out


def test_cli_config_error_is_one_line(capsys, monkeypatch):
    def missing_region():
        raise ConfigError("AWS_REGION is not set")

    monkeypatch.setattr(main, "make_llm", missing_region)
    assert main.main(["hi"]) == 1
    err = capsys.readouterr().err
    assert "AWS_REGION" in err
    assert "Traceback" not in err


def test_cli_aws_error_prints_hint(capsys, monkeypatch):
    def no_creds():
        raise NoCredentialsError()

    monkeypatch.setattr(main, "make_llm", no_creds)
    assert main.main(["hi"]) == 1
    err = capsys.readouterr().err
    assert "AWS_PROFILE" in err
    assert "Bedrock model access" in err
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'main'` (or `app.llm`).

- [ ] **Step 3: Implement `src/app/llm.py`**

```python
"""Bedrock chat model factory, configured from environment variables."""

import os

from langchain_aws import ChatBedrockConverse

# Cross-region inference profile for Claude Sonnet 5.5; confirmed in Task 5.
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-5-5"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def make_llm() -> ChatBedrockConverse:
    region = os.environ.get("AWS_REGION")
    if not region:
        raise ConfigError("AWS_REGION is not set (copy .env.example to .env and fill it in)")
    # No temperature: Sonnet 5.5 rejects non-default sampling parameters.
    return ChatBedrockConverse(
        model=os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID,
        region_name=region,
        credentials_profile_name=os.environ.get("AWS_PROFILE") or None,
        max_tokens=16000,
    )
```

- [ ] **Step 4: Implement `main.py`**

```python
"""CLI entrypoint: python main.py "your prompt"."""

import sys

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from app.graph import build_graph
from app.llm import ConfigError, make_llm

USAGE = 'usage: python main.py "your prompt"'
AWS_HINT = "hint: check AWS_PROFILE, AWS_REGION, and Bedrock model access for BEDROCK_MODEL_ID"


def main(argv: list[str]) -> int:
    prompt = " ".join(argv).strip()
    if not prompt:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        build_graph(make_llm()).invoke({"prompt": prompt})
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    except (BotoCoreError, ClientError) as exc:
        print(f"AWS error: {exc}\n{AWS_HINT}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main(sys.argv[1:]))
```

`load_dotenv()` lives only under `__main__` so a developer's `.env` never leaks into tests.

- [ ] **Step 5: Create `.env.example`**

```
# Required: region where you have Bedrock access to Claude Sonnet 5.5
AWS_REGION=us-east-1
# Optional: named profile from ~/.aws/config (otherwise the default credential chain is used)
AWS_PROFILE=
# Optional: override the model / inference profile ID
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-5-5
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest -v`
Expected: all tests pass. If `llm.model_id` / `llm.region_name` attribute names differ in the installed `langchain-aws`, check with `uv run python -c "from langchain_aws import ChatBedrockConverse; print(list(ChatBedrockConverse.model_fields))"` and fix the test assertions to match the real field names (do not change behavior).

- [ ] **Step 7: Commit**

```bash
git add src/app/llm.py main.py .env.example tests/test_cli.py
git commit -m "feat: add Bedrock client factory and CLI entrypoint" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Live verification against Bedrock and README

Requires the user's AWS credentials with Bedrock access. If none are available in the session, stop and ask the user to run the commands below (they can prefix with `!` in Claude Code) and paste the output.

**Files:**
- Create: `tests/test_live.py`, `README.md` (replace)
- Modify (only if verification requires): `src/app/llm.py` (`DEFAULT_MODEL_ID`), `.env.example`

**Interfaces:**
- Consumes: `app.graph.build_graph`, `app.llm.make_llm`.

- [ ] **Step 1: Write the live test** — `tests/test_live.py`

```python
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
```

- [ ] **Step 2: Confirm the default suite still skips it**

Run: `uv run pytest -v`
Expected: all non-live tests pass; `test_live_end_to_end` is deselected.

- [ ] **Step 3: Find the correct Sonnet 5.5 model ID for the user's region**

Run: `aws bedrock list-inference-profiles --region "$AWS_REGION" --query "inferenceProfileSummaries[?contains(inferenceProfileId, 'sonnet-5-5')].inferenceProfileId"`
Expected: a list containing e.g. `us.anthropic.claude-sonnet-5-5`. If it is empty, run `aws bedrock list-foundation-models --region "$AWS_REGION" --by-provider anthropic --query "modelSummaries[?contains(modelId, 'sonnet-5-5')].modelId"` and use that ID. If the found ID differs from `DEFAULT_MODEL_ID`, update `DEFAULT_MODEL_ID` in `src/app/llm.py` and `BEDROCK_MODEL_ID` in `.env.example` to match.

- [ ] **Step 4: Run the live test**

Run: `cp -n .env.example .env` (then fill in values), then `uv run pytest -m live -v`
Expected: PASS. If it fails with `review failed`, print the raw review reply, tighten `REVIEW_SYSTEM` in `src/app/graph.py`, and re-run until it passes; re-run `uv run pytest -v` afterward so the fake-model tests still pass.

- [ ] **Step 5: Run the CLI manually on both paths**

Run: `uv run python main.py "What is the capital of France?"`
Expected: `=== SUCCESS (confidence …) ===` block.

Run: `uv run python main.py "What will the S&P 500 close at exactly one year from today?"`
Expected: `=== ERROR (confidence …) ===` block (an unknowable question should score ≤ 85).

- [ ] **Step 6: Write `README.md`**

````markdown
# cakesist-langchain-agent

A LangGraph demo on AWS Bedrock (Claude Sonnet 5.5): one node answers a prompt,
a second node reviews the answer and scores its confidence, and the run routes
to a success node (confidence > 85) or an error node (≤ 85).

```
START → generate → review ──(> 85)──→ success → END
                          └─(≤ 85)──→ error   → END
```

## Setup

```bash
uv sync
cp .env.example .env   # set AWS_REGION, optionally AWS_PROFILE / BEDROCK_MODEL_ID
```

Your AWS identity needs Bedrock model access to Claude Sonnet 5.5 in `AWS_REGION`.

## Run

```bash
uv run python main.py "What is the capital of France?"
```

## Test

```bash
uv run pytest          # offline, uses a fake chat model
uv run pytest -m live  # one real Bedrock call
```

The threshold is `CONFIDENCE_THRESHOLD` in `src/app/graph.py`.
````

- [ ] **Step 7: Commit**

```bash
git add tests/test_live.py README.md src/app/llm.py .env.example
git commit -m "test: add live Bedrock check and README" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
