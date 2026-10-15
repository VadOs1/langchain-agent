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


def test_make_llm_unknown_profile_is_config_error(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_PROFILE", "definitely-not-a-real-profile-xyz")
    try:
        make_llm()
    except ConfigError as exc:
        assert "definitely-not-a-real-profile-xyz" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_make_llm_uses_access_keys_with_blank_profile(monkeypatch):
    # Mirrors a .env with `AWS_PROFILE=` left blank and static keys filled in.
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_PROFILE", "")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTDUMMY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    creds = make_llm().client._request_signer._credentials
    assert creds.access_key == "AKIATESTDUMMY"
    assert creds.secret_key == "dummy-secret"
