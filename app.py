# app.py
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage
import rag_setup
import agent_workflow

from agent_workflow import AgentState, invoke_app, MEDICAL_DISCLAIMER
from logging_setup import SYSTEM_LOGGER
from tools import get_patient_discharge_report

st.set_page_config(
    page_title="DataSmith AI - Post-Discharge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: white; text-align: center; }
    .main-header h1 { color: white; font-size: 2.5rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
    .main-header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 0.5rem; }
    .medical-disclaimer { background: linear-gradient(135deg,#ffecd2 0%,#fcb69f 100%); padding:1.5rem; border-radius:10px; border-left:5px solid #ff6b6b; margin-bottom:2rem; box-shadow:0 2px 4px rgba(0,0,0,0.1); }
    .medical-disclaimer strong { color:#c92a2a; font-size:1.1rem; }
    .stChatMessage { padding:1rem; border-radius:15px; margin-bottom:1rem; }
    [data-testid="stChatMessage"][aria-label*="user"] { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color: white; }
    [data-testid="stChatMessage"][aria-label*="assistant"] { background: linear-gradient(135deg,#f093fb 0%,#f5576c 100%); color: white; }
    .stChatInputContainer { background: white; border-radius:25px; padding:1rem; box-shadow:0 4px 6px rgba(0,0,0,0.1); }
    .css-1d391kg { background: linear-gradient(180deg,#667eea 0%,#764ba2 100%); }
    .stButton>button { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; border:none; border-radius:20px; padding:0.5rem 2rem; font-weight:bold; transition:all .3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow:0 4px 8px rgba(0,0,0,0.2); }
    .streamlit-expanderHeader { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; border-radius:10px; padding:1rem; }
    .status-indicator { display:inline-block; width:12px; height:12px; border-radius:50%; background:#51cf66; margin-right:8px; animation:pulse 2s infinite; }
    @keyframes pulse { 0%{opacity:1;}50%{opacity:0.5;}100%{opacity:1;} }
    .info-card { background:white; padding:1.5rem; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1); margin-bottom:1rem; border-left:4px solid #667eea; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-header">
        <h1>🏥 DataSmith AI Post-Discharge Medical Assistant</h1>
        <p>Your trusted companion for post-discharge care and medical guidance</p>
        <p style="font-size:0.9rem;margin-top:0.5rem;">
            <span class="status-indicator"></span> System Online • Ready to Assist
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="medical-disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong><br>{MEDICAL_DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 Nephrology Reference (RAG)")
    st.caption("Upload a large nephrology PDF (optional) to use as the internal RAG reference.")
    uploaded_ref = st.file_uploader("Upload nephrology_reference.pdf (optional)", type=["pdf"], key="ref_upload")

    if uploaded_ref is not None:
        ref_path = "nephrology_reference.pdf"
        with open(ref_path, "wb") as f:
            f.write(uploaded_ref.getbuffer())
        st.success(f"Reference uploaded: {ref_path}")

    st.markdown("**Rebuild/Load RAG DB**")
    if st.button("Rebuild RAG (create embeddings)", key="rebuild_rag"):
        try:
            with st.spinner("Building or loading RAG vectorstore (this may take several minutes)..."):
                chain = rag_setup.setup_rag_retriever(
                    source_file=None,
                    persist_directory=rag_setup.CHROMA_DB_PATH,
                    rebuild=True
                )
                agent_workflow.RAG_RETRIEVAL_CHAIN = chain
            st.success("RAG vector store created and loaded.")
        except Exception as e:
            st.error(f"Failed to build RAG: {e}")
            SYSTEM_LOGGER.exception("RAG build error")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {
        "messages": [],
        "current_agent": "Receptionist Agent",
        "patient_report": ""
    }
    st.session_state["messages"].append({
        "role": "assistant",
        "content": (
            "👋 Hello! I'm your post-discharge care assistant. I'm here to help with questions about "
            "your medications, discharge instructions, and general health guidance.\n\n**To get started, please tell me your name.**"
        )
    })

st.markdown("### 💬 Conversation")
chat_container = st.container()
with chat_container:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Type your message here... (e.g., 'My name is John Smith' or 'What are my medications?')")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state["agent_state"]["messages"].append(HumanMessage(content=prompt))

    current_agent = st.session_state["agent_state"].get("current_agent", "Receptionist Agent")
    agent_emoji = "👨‍⚕️" if "Clinical" in current_agent else "👋"

    with st.spinner(f"{agent_emoji} {current_agent} is processing your request..."):
        try:
            final_state = invoke_app(st.session_state["agent_state"])
        except Exception as e:
            SYSTEM_LOGGER.exception("Failed when invoking agent graph.")
            st.error(f"❌ Agent error: {e}")
            final_state = st.session_state["agent_state"]

        assistant_response = "I'm sorry, I didn't receive a response. Please try again."
        try:
            messages = final_state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    assistant_response = last_msg.content
                elif isinstance(last_msg, dict):
                    assistant_response = last_msg.get("content", str(last_msg))
                else:
                    assistant_response = str(last_msg)
        except Exception as e:
            SYSTEM_LOGGER.exception(f"Error extracting assistant response: {e}")
            assistant_response = "I encountered an error processing your request. Please try again."

        st.session_state["agent_state"] = final_state
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        **🔒 Privacy & Security**
        - Your data is secure
        - HIPAA-compliant practices
        - No data stored permanently
        """
    )

with col2:
    st.markdown(
        """
        **🤖 AI Assistant**
        - Powered by advanced AI
        - Clinical reference materials
        - Real-time web search capability
        """
    )

with col3:
    st.markdown(
        """
        **📊 System Status**
        - Multi-agent system active
        - RAG system operational
        - Logging enabled
        """
    )

with st.expander("🔍 System Logs & Diagnostics", expanded=False):
    try:
        with open("system_logs.log", "r", encoding="utf-8") as f:
            log_content = f.read()
            if log_content:
                st.code(log_content, language="text")
            else:
                st.info("No logs generated yet. System is ready.")
    except FileNotFoundError:
        st.info("System logs not yet generated.")
    except Exception as e:
        st.warning(f"Could not read logs: {e}")

st.markdown(
    """
    <div style="text-align:center;color:#666;padding:1rem;">
        <p><strong>💡 Tip:</strong> Be specific with your questions for the best responses. You can ask about medications, dietary restrictions, warning signs, or follow-up appointments.</p>
    </div>
    """,
    unsafe_allow_html=True
)
