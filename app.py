import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from core.graph import support_app

st.set_page_config(
    page_title="VoltPro Diagnostic Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ---- Base ---- */
    .stApp { background-color: #0d1117; color: #f0f6fc; }
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #30363d !important; }

    /* ---- Chat bubbles ---- */
    .stChatMessage { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 10px !important; padding: 14px !important; margin-bottom: 12px !important; }
    .stChatMessage p, .stChatMessage div, .stChatMessage span, .stChatMessage li { color: #f0f6fc !important; font-size: 15px !important; line-height: 1.65 !important; }
    .stChatMessage strong { color: #58a6ff !important; }
    .stChatMessage code { background-color: #0d1117 !important; color: #79c0ff !important; border-radius: 4px !important; padding: 1px 5px !important; }

    /* ---- Chat input (targeted explicitly — this is a native Streamlit
       widget, not covered by the generic rules above, and is what goes
       invisible when the device is in light mode without this override) ---- */
    [data-testid="stChatInput"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f0f6fc !important;
        background-color: transparent !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8b949e !important;
        opacity: 1 !important;
    }

    /* ---- Buttons ---- */
    .stButton>button { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 8px; font-weight: 500; transition: all 0.2s ease; }
    .stButton>button:hover { background-color: #30363d; color: #58a6ff; border-color: #58a6ff; }

    /* ---- Radio (session list) — also native, also needs explicit text color ---- */
    [data-testid="stSidebar"] label p { color: #c9d1d9 !important; }

    /* ---- Status / expander widgets ---- */
    [data-testid="stExpander"] { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

if "sessions_meta" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.sessions_meta = {initial_id: "New Diagnostic Session"}
    st.session_state.current_thread_id = initial_id

with st.sidebar:
    st.markdown("### Diagnostic Console")
    st.caption("Industrial Equipment & Microgrid Troubleshooting")

    if st.button("New Session", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.sessions_meta[new_id] = "New Diagnostic Session"
        st.session_state.current_thread_id = new_id
        st.rerun()

    st.write("---")
    st.markdown("##### Previous Sessions")

    session_keys = list(st.session_state.sessions_meta.keys())
    session_titles = [st.session_state.sessions_meta[k] for k in session_keys]
    current_index = session_keys.index(st.session_state.current_thread_id)
    selected_title = st.radio("Select Session:", session_titles, index=current_index, label_visibility="collapsed")

    selected_key = session_keys[session_titles.index(selected_title)]
    if selected_key != st.session_state.current_thread_id:
        st.session_state.current_thread_id = selected_key
        st.rerun()

    st.write("---")
    st.markdown("##### Quick Diagnostic Scenarios")
    quick_query = None
    if st.button("Thermal Overload E-204", use_container_width=True):
        quick_query = "Inverter shows Error E-204 with loud fans. What is the root cause and which part should I clean?"
    if st.button("M8 Battery Lugs Torque", use_container_width=True):
        quick_query = "What is the exact torque specification for battery M8 connection lugs?"
    if st.button("Telemetry: Unit SN-9002", use_container_width=True):
        quick_query = "Check online telemetry and active alarms for serial SN-9002."
    if st.button("Spare Part: FAN-4412", use_container_width=True):
        quick_query = "Check warehouse stock and price for spare part FAN-4412."

    st.write("---")
    st.markdown("##### System Status")
    st.success("LangGraph Engine: Active")
    st.info("Qdrant Vector Store: Connected")
    st.caption(f"Thread: {st.session_state.current_thread_id[:18]}...")

st.title("⚡ VoltPro Diagnostic Copilot")
st.caption("Agentic hardware diagnostics — manual RAG, live telemetry, and parts lookup")

config = {
    "configurable": {"thread_id": st.session_state.current_thread_id},
    "recursion_limit": 25,
}
current_state = support_app.get_state(config)

if current_state and "messages" in current_state.values:
    for msg in current_state.values["messages"]:
        if msg.type in ["human", "ai"] and msg.content:
            role = "user" if msg.type == "human" else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

user_input = quick_query or st.chat_input("Ask a technical question, enter an error code (e.g. E-204), or serial number...")

if user_input:
    if st.session_state.sessions_meta[st.session_state.current_thread_id] == "New Diagnostic Session":
        cleaned_title = user_input[:26] + ("..." if len(user_input) > 26 else "")
        st.session_state.sessions_meta[st.session_state.current_thread_id] = cleaned_title

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("Analyzing telemetry & technical documentation...", expanded=True) as status_box:
            st.write("Routing through LangGraph decision nodes...")
            response = support_app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            status_box.update(label="Diagnostic analysis complete", state="complete", expanded=False)

        final_answer = response["messages"][-1].content
        st.markdown(final_answer)

        tool_messages = [m for m in response["messages"] if m.type == "tool"]
        if tool_messages:
            with st.expander("View Technical Evidence & Retrieved Citations"):
                for tm in tool_messages:
                    st.code(tm.content, language="markdown")

    st.rerun()
