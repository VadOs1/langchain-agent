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
