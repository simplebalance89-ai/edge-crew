import os
from openai import OpenAI


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _normalize_endpoint(url: str | None) -> str:
    if not url:
        return ""
    return url if url.endswith("/") else url + "/"


AZURE_AI_ENDPOINT = _normalize_endpoint(_first_env(
    "AZURE_AI_ENDPOINT",
    "AZURE_INFERENCE_ENDPOINT",
    "AZURE_FOUNDRY_ENDPOINT",
))
AZURE_OPENAI_ENDPOINT = _normalize_endpoint(_first_env("AZURE_OPENAI_ENDPOINT"))

AZURE_AI_KEY = _first_env(
    "AZURE_AI_KEY",
    "AZURE_INFERENCE_KEY",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_API_KEY",
)
AZURE_OPENAI_KEY = _first_env(
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_KEY",
    "AZURE_AI_KEY",
    "AZURE_INFERENCE_KEY",
)
AZURE_OPENAI_API_VERSION = _first_env("AZURE_OPENAI_API_VERSION", default="2024-12-01-preview")

MODEL_SPECS = {
    "grok": {
        "deployment": _first_env("AZURE_GROK_DEPLOYMENT", default="grok-3"),
        "endpoint_type": "ai_services",
    },
    "grok_fast": {
        "deployment": _first_env("AZURE_GROK_FAST_DEPLOYMENT", default="grok-4-1-fast-reasoning"),
        "endpoint_type": "ai_services",
    },
    "grok-fast": {
        "deployment": _first_env("AZURE_GROK_FAST_DEPLOYMENT", default="grok-4-1-fast-reasoning"),
        "endpoint_type": "ai_services",
    },
    "deepseek": {
        "deployment": _first_env("AZURE_DEEPSEEK_DEPLOYMENT", default="DeepSeek-V3.2"),
        "endpoint_type": "ai_services",
    },
    "gpt41": {
        "deployment": _first_env("AZURE_GPT41_DEPLOYMENT", default="gpt-4.1"),
        "endpoint_type": "openai",
    },
    "gpt41mini": {
        "deployment": _first_env("AZURE_GPT41MINI_DEPLOYMENT", default="gpt-4.1-mini"),
        "endpoint_type": "openai",
    },
}


def get_model_spec(model_key: str) -> dict:
    return MODEL_SPECS.get(model_key, MODEL_SPECS["grok"])


def build_client(model_key: str) -> tuple[OpenAI, dict]:
    spec = get_model_spec(model_key)
    deployment = spec["deployment"]
    endpoint_type = spec["endpoint_type"]

    if endpoint_type == "openai":
        client = OpenAI(
            base_url=f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{deployment}/",
            api_key=AZURE_OPENAI_KEY,
            default_headers={"api-key": AZURE_OPENAI_KEY},
            default_query={"api-version": AZURE_OPENAI_API_VERSION},
        )
    else:
        client = OpenAI(
            base_url=AZURE_AI_ENDPOINT,
            api_key=AZURE_AI_KEY,
        )

    return client, spec
