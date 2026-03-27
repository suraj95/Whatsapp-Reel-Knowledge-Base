import os


def _read_int_env(name: str, default: int, minimum: int = 1, maximum: int = 100) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


# Controls how many Pinecone matches are requested per chat query.
# Override with UI_QUERY_TOP_K in .env for easy tuning.
CHAT_TOP_K = _read_int_env("UI_QUERY_TOP_K", default=3, minimum=1, maximum=20)
