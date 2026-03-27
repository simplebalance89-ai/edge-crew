import os

PROXY_ENV_VARS = (
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
)

DEAD_PROXY_MARKERS = (
    "127.0.0.1:9",
    "localhost:9",
)


def first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def require_env(name: str, service: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    raise RuntimeError(f"{service} requires the `{name}` environment variable")


def sanitized_env(env: dict[str, str] | None = None) -> dict[str, str]:
    cleaned = dict(env or os.environ)
    for name in PROXY_ENV_VARS:
        value = cleaned.get(name, "")
        if any(marker in value for marker in DEAD_PROXY_MARKERS):
            cleaned.pop(name, None)
    return cleaned


def remove_dead_local_proxy_env() -> list[str]:
    removed: list[str] = []
    for name in PROXY_ENV_VARS:
        value = os.environ.get(name, "")
        if any(marker in value for marker in DEAD_PROXY_MARKERS):
            os.environ.pop(name, None)
            removed.append(name)
    return removed
