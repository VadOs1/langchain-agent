"""Bedrock chat model factory, configured from environment variables."""

import os

from langchain_aws import ChatBedrockConverse
from pydantic import ValidationError

DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def make_llm() -> ChatBedrockConverse:
    region = os.environ.get("AWS_REGION")
    if not region:
        raise ConfigError("AWS_REGION is not set (copy .env.example to .env and fill it in)")
    # A blank `AWS_PROFILE=` from .env makes boto3 look up a profile named "",
    # which fails before AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are tried.
    if not os.environ.get("AWS_PROFILE"):
        os.environ.pop("AWS_PROFILE", None)
    try:
        return ChatBedrockConverse(
            model=os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID,
            region_name=region,
            credentials_profile_name=os.environ.get("AWS_PROFILE") or None,
            max_tokens=16000,
        )
    except ValidationError as exc:
        # langchain-aws turns client setup failures (e.g. unknown AWS_PROFILE) into
        # a pydantic ValidationError rather than a botocore error.
        raise ConfigError(exc.errors()[0]["msg"]) from exc
