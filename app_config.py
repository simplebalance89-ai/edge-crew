import os


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
