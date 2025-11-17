# app.py
import streamlit as st
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
# RAG / agent control imports (add near other imports)
import rag_setup
import agent_workflow
import os


# Import project modules
from agent_workflow import AgentState, invoke_app, MEDICAL_DISCLAIMER
from logging_setup import SYSTEM_LOGGER
from tools import get_patient_discharge_report

# Page configuration
st.set_page_config(
    page_title="DataSmith AI - Post-Discharge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for medical theme
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Medical disclaimer box */
    .medical-disclaimer {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .medical-disclaimer strong {
        color: #c92a2a;
        font-size: 1.1rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    
    /* User message styling */
    [data-testid="stChatMessage"][aria-label*="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"][aria-label*="assistant"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        background: white;
        border-radius: 25px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #51cf66;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Card styling for info boxes */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="main-header">
        <h1>🏥 DataSmith AI Post-Discharge Medical Assistant</h1>
        <p>Your trusted companion for post-discharge care and medical guidance</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">
            <span class="status-indicator"></span>
            System Online • Ready to Assist
        </p>
    </div>
""", unsafe_allow_html=True)

# Medical Disclaimer
st.markdown(f"""
    <div class="medical-disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong><br>
        {MEDICAL_DISCLAIMER}
    </div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 Nephrology Reference (RAG)")
    st.caption("Upload a large nephrology PDF (e.g., 1500 pages) to use as the internal reference for RAG.")
    uploaded_ref = st.file_uploader("Upload nephrology_reference.pdf (optional)", type=["pdf"], key="ref_upload")

    if uploaded_ref is not None:
        # Save uploaded file to disk for ingestion
        ref_path = "nephrology_reference.pdf"
        with open(ref_path, "wb") as f:
            f.write(uploaded_ref.getbuffer())
        st.success(f"Reference uploaded: {ref_path}")

    st.markdown("**Rebuild/Load RAG DB**")
    rebuild_rag = st.button("Rebuild RAG (create embeddings)", key="rebuild_rag")
    if rebuild_rag:
        import rag_setup
        try:
            with st.spinner("Building or loading RAG vectorstore (this may take several minutes)..."):
                # If the uploaded pdf exists it will be used by default. Rebuild forces recreate.
                chain = rag_setup.setup_rag_retriever(source_file=None, persist_directory=rag_setup.CHROMA_DB_PATH, rebuild=True)
                # update global reference so rag_query_tool uses it
                import agent_workflow
                agent_workflow.RAG_RETRIEVAL_CHAIN = chain
            st.success("RAG vector store created and loaded.")
        except Exception as e:
            st.error(f"Failed to build RAG: {e}")
            import logging
            logging.exception("RAG build error")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {
        "messages": [],
        "current_agent": "Receptionist Agent",
        "patient_report": ""
    }
    # Initial greeting
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "👋 Hello! I'm your post-discharge care assistant. I'm here to help you with questions about your medications, discharge instructions, and general health guidance.\n\n**To get started, please tell me your name.**"
    })

# Main chat container
st.markdown("### 💬 Conversation")

# Display messages with enhanced styling
chat_container = st.container()
with chat_container:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle input with enhanced styling
if prompt := st.chat_input("Type your message here... (e.g., 'My name is John Smith' or 'What are my medications?')"):
    # Append user message to UI history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Append to agent_state messages as a HumanMessage
    st.session_state["agent_state"]["messages"].append(HumanMessage(content=prompt))

    # Show processing indicator
    current_agent = st.session_state['agent_state']['current_agent']
    agent_emoji = "👨‍⚕️" if "Clinical" in current_agent else "👋"
    
    with st.spinner(f"{agent_emoji} {current_agent} is processing your request..."):
        try:
            final_state = invoke_app(st.session_state["agent_state"])
        except Exception as e:
            SYSTEM_LOGGER.exception("Failed when invoking agent graph.")
            st.error(f"❌ Agent error: {e}")
            final_state = st.session_state["agent_state"]

        # Extract the assistant's response from the final state
        assistant_response = None
        try:
            messages = final_state.get("messages", [])
            if messages:
                # Get the last message (should be the AI response)
                last_msg = messages[-1]
                # Extract content based on message type
                if hasattr(last_msg, "content"):
                    assistant_response = last_msg.content
                elif isinstance(last_msg, dict):
                    assistant_response = last_msg.get("content", str(last_msg))
                else:
                    assistant_response = str(last_msg)
            else:
                assistant_response = "I'm sorry, I didn't receive a response. Please try again."
        except Exception as e:
            SYSTEM_LOGGER.exception(f"Error extracting assistant response: {e}")
            assistant_response = "I encountered an error processing your request. Please try again."

        # Update session state
        st.session_state["agent_state"] = final_state
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# Footer with system information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🔒 Privacy & Security**
    - Your data is secure
    - HIPAA-compliant practices
    - No data stored permanently
    """)

with col2:
    st.markdown("""
    **🤖 AI Assistant**
    - Powered by advanced AI
    - Clinical reference materials
    - Real-time web search capability
    """)

with col3:
    st.markdown("""
    **📊 System Status**
    - Multi-agent system active
    - RAG system operational
    - Logging enabled
    """)

# System Logs (collapsible)
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

# Add some helpful tips at the bottom
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>💡 Tip:</strong> Be specific with your questions for the best responses. 
    You can ask about medications, dietary restrictions, warning signs, or follow-up appointments.</p>
</div>
""", unsafe_allow_html=True)
