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
