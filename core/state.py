from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state container passed across all LangGraph nodes."""
    messages: Annotated[list, add_messages]
