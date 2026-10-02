"""
Centralized telemetry and observability module for VoltPro Diagnostic Copilot.
Supports automated LangSmith tracing and Langfuse distributed callbacks.
"""

import os
from typing import List, Any


def setup_observability() -> None:
    """Configures global environment flags for automated LangSmith tracing."""
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "voltpro-copilot"))

    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        os.environ.setdefault("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")


def get_telemetry_callbacks() -> List[Any]:
    """Collects runtime telemetry handlers for LangChain and LangGraph."""
    callbacks = []

    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.langchain import CallbackHandler
            callbacks.append(CallbackHandler())
        except Exception:
            pass

    return callbacks


def flush_telemetry() -> None:
    """Flushes telemetry buffers to ensure all events reach the cloud."""
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import get_client
            get_client().flush()
        except Exception:
            pass
