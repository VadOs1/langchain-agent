# Review node: native JSON-schema structured output

Status: proposed, not implemented.

## Goal

Replace the prompt-instructed JSON reply in the review node with Bedrock's
native structured output (`outputConfig.textFormat` of type `json_schema`), the
same Claude feature the Agent SDK exposes as `outputFormat: {type: 'json_schema'}`.
The model's reply is then constrained to the `ReviewResult` schema, so the
brace-scanning parser and the "reply with only JSON" prompt wording go away.

## Background

The original design (`2026-10-14-bedrock-review-graph-design.md`, "Sonnet 5.5
constraints" §1) rejected `with_structured_output` because its default method,
`function_calling`, forces `tool_choice`, which Sonnet 5.5 rejects with HTTP 400.

`langchain-aws` 1.8.2 (locked) also offers
`ChatBedrockConverse.with_structured_output(schema, method="json_schema")`
(`langchain_aws/chat_models/bedrock_converse.py`, `_with_structured_output_json_schema`).
It binds `output_config` instead of tools, so `tool_choice` is never sent and that
constraint does not apply. It parses the reply with `PydanticOutputParser` for
Pydantic schemas, and supports `include_raw=True`.

## Open question to settle first (live check)

Claude structured outputs do not support numeric constraints (`minimum`,
`maximum`). The Anthropic SDK strips them; `langchain-aws` does not — it only
adds `additionalProperties: false` and forwards the rest. `ReviewResult`
currently emits `"minimum": 0, "maximum": 100` via `Field(ge=0, le=100)`.

Before implementing, run one live call with each of:

1. `ReviewResult` as-is (with `ge`/`le`), and
2. an otherwise identical model with plain `confidence: int`.

Record whether (1) returns HTTP 400, or succeeds (constraints ignored). Either
way, the design below sends the unconstrained schema, so the result only decides
whether that split is required or merely defensive. Note the outcome here.

Probe sketch (run from the repo root with AWS credentials configured):

```python
from dotenv import load_dotenv; load_dotenv(".env")
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm import make_llm
from app.graph import ReviewResult

class Unconstrained(BaseModel):
    confidence: int
    reasoning: str

msgs = [SystemMessage("Rate 0-100 how confident you are the answer is correct."),
        HumanMessage("Question:\nCapital of France?\n\nProposed answer:\nLyon.")]
llm = make_llm()
for schema in (ReviewResult, Unconstrained):
    try:
        r = llm.with_structured_output(schema, method="json_schema", include_raw=True).invoke(msgs)
        print(schema.__name__, r["parsed"], r["parsing_error"])
    except Exception as e:
        print(schema.__name__, "FAILED:", type(e).__name__, str(e)[:400])
```

Also confirm the reply still parses with adaptive thinking on (reasoning blocks
present alongside the JSON text block).

## Design

### Schema

- Add a model-facing schema with no numeric constraints, e.g. `ReviewOutput`
  (`confidence: int`, `reasoning: str`), used only for `outputConfig`.
- Keep `ReviewResult` (`Field(ge=0, le=100)`) as the validated domain type.
  Validate the parsed `ReviewOutput` into `ReviewResult` so out-of-range values
  still fail and route to `error`.

### Prompt

Drop the format instructions; the schema enforces shape. Proposed:

```python
REVIEW_SYSTEM = (
    "You are a strict reviewer. Given a question and a proposed answer, judge how "
    "confident you are (0-100) that the answer is correct, complete, and directly "
    "addresses the question, and explain your judgment in one or two sentences."
)
```

The 0-100 range stays in the prompt because it is no longer in the schema.

### Graph wiring

`build_graph(llm)` currently takes only a `BaseChatModel`, and the tests use
`GenericFakeChatModel`, which does not implement `with_structured_output`.
Keep the tests model-agnostic by injecting the reviewer:

- `build_graph(llm, reviewer=None)`, where `reviewer` is a runnable taking the
  review messages and returning a `ReviewOutput` (or dict).
- Default (`reviewer is None`): `llm.with_structured_output(ReviewOutput, method="json_schema")`.
- The review node invokes the reviewer, then `ReviewResult.model_validate(...)`.
  Any exception from the call's parsing step or validation (`OutputParserException`,
  `ValidationError`, `ValueError`) yields `confidence=0`,
  `review_reasoning="review failed: <first line>"` — same behavior as today.
  AWS/Bedrock errors are not caught here; they keep propagating to the CLI as now.

### Removals

- `parse_review` and `_DECODER` in `src/app/graph.py`.
- `message_text` stays (the generate node still uses it).

## Tests

- `tests/test_review_parsing.py`: drop `parse_review` tests; keep `message_text`
  tests. Add `ReviewResult` validation tests for out-of-range/missing values if
  not covered elsewhere.
- `tests/test_graph.py`: pass a fake reviewer (e.g. `RunnableLambda` returning a
  scripted `ReviewOutput`/dict, or raising) alongside the fake generate model.
  Cover: high confidence → success; low → error; 85 → error; reviewer raises a
  parse error → error with `confidence == 0`; out-of-range (150) → error.
- Add a unit test that the default reviewer binds `output_config` with
  `textFormat.type == "json_schema"` and that the sent schema has no
  `minimum`/`maximum` (inspect the bound runnable's kwargs; no network).
- `tests/test_live.py`: extend the live check to exercise the review node end to
  end and assert a parsed confidence in 0-100.

## Docs

- Update the original design spec's "Sonnet 5.5 constraints" §1 to say forced
  `tool_choice` is still avoided, and that review uses `method="json_schema"`.
- Note the live-check outcome (open question above) in this file.

## Out of scope

Changing the generate node, the threshold, or the model.
