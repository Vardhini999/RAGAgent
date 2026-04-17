"""
Langfuse observability wrapper.
If LANGFUSE_PUBLIC_KEY is not set, all calls are no-ops — zero friction for local dev.
"""
from typing import Any

from src.config import settings

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not settings.langfuse_public_key:
        return None
    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except ImportError:
        pass
    return _client


class Trace:
    """Context manager wrapping a Langfuse trace. No-op when Langfuse is disabled."""

    def __init__(self, name: str, metadata: dict | None = None):
        self.name = name
        self.metadata = metadata or {}
        self._trace = None

    def __enter__(self):
        client = _get_client()
        if client:
            self._trace = client.trace(name=self.name, metadata=self.metadata)
        return self

    def __exit__(self, *_):
        if self._trace:
            self._trace.update(output=self.metadata.get("output"))

    def span(self, name: str, input: Any = None, output: Any = None, metadata: dict | None = None):
        if self._trace:
            self._trace.span(name=name, input=str(input), output=str(output), metadata=metadata)

    def score(self, name: str, value: float, comment: str = ""):
        if self._trace:
            self._trace.score(name=name, value=value, comment=comment)

    @property
    def id(self) -> str:
        return self._trace.id if self._trace else ""
