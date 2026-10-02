"""
End-to-end deterministic control-flow tests for VoltPro Diagnostic Copilot.
Runs using a ScriptedLLM stand-in: fully reproducible, zero API keys, zero network.
Tests state graph transitions, tool dispatch, and conversation summarization.
"""

import itertools
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from core.graph import build_graph, MAX_MESSAGES_BEFORE_SUMMARY

_ids = itertools.count(1)


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


class ScriptedLLM:
    """Deterministic stand-in returning pre-scripted messages in order."""

    def __init__(self, script):
        self.script = list(script)
        self.tools_offered = []
        self.system_prompts = []

    def bind_tools(self, tools):
        names = [t.name for t in tools]
        outer = self

        class _Bound:
            def invoke(self, messages, *args, **kwargs):
                outer.tools_offered.append(names)
                outer.system_prompts.append(messages[0].content)
                return outer.script.pop(0)

        return _Bound()

    def invoke(self, messages, *args, **kwargs):
        return self.script.pop(0)


def run(llm, question, thread="voltpro_test"):
    app = build_graph(llm=llm, checkpointer=MemorySaver())
    return app, app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread}, "recursion_limit": 25},
    )


def ai_texts(state):
    return [m.content for m in state["messages"] if m.type == "ai" and m.content]


def test_scope_refusal_when_unrelated_topic():
    llm = ScriptedLLM([AIMessage(content="I am only authorized to support VoltPro-X9000 inverter topics.")])
    _, state = run(llm, "How do I make chocolate cake?")
    assert "only authorized" in ai_texts(state)[-1]
    assert len(llm.tools_offered) == 1


def test_telemetry_routing_for_serial_number():
    llm = ScriptedLLM([
        call("lookup_device_telemetry", serial_number="SN-9002"),
        AIMessage(content="Unit SN-9002 is currently experiencing Error E-204 thermal overload at 97C.")
    ])
    _, state = run(llm, "Check status for serial SN-9002")
    assert any(m.name == "lookup_device_telemetry" for m in state["messages"] if m.type == "tool")
    assert "SN-9002" in ai_texts(state)[-1]


def test_spare_parts_routing_for_part_number():
    llm = ScriptedLLM([
        call("lookup_spare_part", part_number="FAN-4412"),
        AIMessage(content="Part FAN-4412 is in stock with 12 units available at $65.00.")
    ])
    _, state = run(llm, "Do we have FAN-4412 in stock?")
    assert any(m.name == "lookup_spare_part" for m in state["messages"] if m.type == "tool")
    assert "$65.00" in ai_texts(state)[-1]


def test_manual_search_routing_for_error_code():
    llm = ScriptedLLM([
        call("search_technical_manual", query="Error E-204 remediation"),
        AIMessage(content="[Source: Section 3, E-204] Clean bottom air filter AF-901 and check fan assembly FAN-4412.")
    ])
    _, state = run(llm, "What should I do for Error E-204?")
    assert any(m.name == "search_technical_manual" for m in state["messages"] if m.type == "tool")
    assert "[Source: Section 3" in ai_texts(state)[-1]


def test_summarization_triggers_after_threshold():
    script = [
        AIMessage(content="This is a summary of previous discussion regarding SN-9002 and Error E-204."),
        AIMessage(content="Final response.")
    ]
    llm = ScriptedLLM(script)
    app = build_graph(llm=llm, checkpointer=MemorySaver())

    messages = [HumanMessage(content=f"Message {i}") for i in range(13)]
    state = app.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": "summary_thread"}, "recursion_limit": 25}
    )

    system_summaries = [
        m for m in state["messages"]
        if isinstance(m, SystemMessage) and "[Earlier conversation summary]" in m.content
    ]
    assert len(system_summaries) == 1
    assert "SN-9002" in system_summaries[0].content or "E-204" in system_summaries[0].content


def test_tool_failure_does_not_crash_graph():
    llm = ScriptedLLM([
        call("lookup_device_telemetry", serial_number="UNKNOWN-9999"),
        AIMessage(content="The serial number was not found in the live telemetry network.")
    ])
    _, state = run(llm, "Check UNKNOWN-9999")
    assert any(m.type == "tool" for m in state["messages"])
    assert "not found" in ai_texts(state)[-1]
