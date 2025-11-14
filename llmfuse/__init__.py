"""Top-level package for the LLM-backed filesystem."""

__all__ = ["LLMFuse", "FSState", "FileEntry"]


def __getattr__(name):
    if name == "LLMFuse":
        from .llmfuse import LLMFuse as _LLMFuse

        return _LLMFuse
    if name == "FSState":
        from .fs_state import FSState as _FSState

        return _FSState
    if name == "FileEntry":
        from .fs_state import FileEntry as _FileEntry

        return _FileEntry
    raise AttributeError(f"module 'llmfuse' has no attribute {name!r}")
