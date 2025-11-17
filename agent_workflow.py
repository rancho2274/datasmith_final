from typing import TypedDict, Annotated, List
import operator
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Import agent creation functions - use LangChain 1.0+ API
try:
    # Try the new LangChain 1.0+ API first
    from langchain.agents import create_agent
    USE_NEW_API = True
except ImportError:
    try:
        # Try the standard import (older versions)
        from langchain.agents import create_react_agent, AgentExecutor
        USE_NEW_API = False
    except ImportError:
        try:
            # Try alternative import path
            from langchain.agents.react.agent import create_react_agent
            from langchain.agents import AgentExecutor
            USE_NEW_API = False
        except ImportError:
            # Use the OpenAI tools agent API
            try:
                from langchain.agents import AgentExecutor, create_openai_tools_agent
                from langchain_core.prompts import MessagesPlaceholder
                USE_NEW_API = False
                
                def create_react_agent(llm, tools, prompt):
                    """Create a ReAct-style agent using OpenAI tools agent"""
                    return create_openai_tools_agent(llm, tools, prompt)
            except ImportError:
                raise ImportError(
                    "Could not import agent creation functions. Please ensure langchain and langchain-openai are installed. "
                    "Try: pip install langchain langchain-openai langchain-community"
                )

from logging_setup import SYSTEM_LOGGER
from tools import get_patient_discharge_report, clinical_web_search
from rag_setup import RAG_RETRIEVAL_CHAIN

# RAG_RETRIEVAL_CHAIN will be created by rag_setup.setup_rag_retriever()
# You can import it after you have executed setup_rag_retriever and created variable.

# Define the shared state for the entire graph
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    'messages' is a list of all messages (user input, agent output, tool calls).
    'current_agent' tracks which agent has control (used for conditional routing).
    'patient_report' stores the retrieved discharge data.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    current_agent: str
    patient_report: str

# Initialize LLM
LLM = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create RAG tool wrapper for the clinical agent
from langchain_core.tools import tool

@tool
def rag_query_tool(query: str) -> str:
    """
    Queries the nephrology reference book using RAG (Retrieval Augmented Generation).
    Returns relevant information with source citations from the reference materials.
    Use this tool first for medical questions before using web search.
    """
    if RAG_RETRIEVAL_CHAIN is None:
        return "ERROR: RAG system not initialized. Please check the setup."
    
    SYSTEM_LOGGER.info(f"RAG QUERY: Searching reference materials for: '{query}'")
    try:
        result = RAG_RETRIEVAL_CHAIN.invoke({"input": query})
        rag_output = result.get("output", "No relevant information found.")
        SYSTEM_LOGGER.info("RAG QUERY SUCCESS: Retrieved relevant context with citations.")
        # The RAG chain already includes citations, so we just return it
        return rag_output
    except Exception as e:
        SYSTEM_LOGGER.error(f"RAG QUERY ERROR: {e}")
        return f"ERROR: Unable to query reference materials. Error: {e}"

# 1. Gather all tools for the respective agents
RECEPTIONIST_TOOLS = [get_patient_discharge_report] 
# The Clinical Agent uses RAG tool first, then web search as fallback
CLINICAL_TOOLS = [rag_query_tool, clinical_web_search]

# --- LLM and Prompts ---
# IMPORTANT: Include medical disclaimers in the main system prompts (Requirement 7.1)
MEDICAL_DISCLAIMER = (
    "NOTE: This is an AI assistant for educational purposes only. "
    "Always consult healthcare professionals for medical advice."
)

# Create system prompts for agents
RECEPTIONIST_SYSTEM_PROMPT = (
    f"You are the Post-Discharge Receptionist AI. Your primary goal is to retrieve the patient's report and route medical queries. {MEDICAL_DISCLAIMER}\n\n"
    "**Detailed Workflow:**\n"
    "1. **Greet the patient** and ask for their full name.\n"
    "2. **Retrieve the report**: Use the `get_patient_discharge_report` tool ONLY ONCE when you have the patient's name. "
    "After retrieving the report, acknowledge that you found their information.\n"
    "3. **Ask follow-up questions**: After retrieving the discharge report, proactively ask helpful follow-up questions such as:\n"
    "   - 'Would you like to know about your medications?'\n"
    "   - 'Do you have questions about your dietary restrictions?'\n"
    "   - 'Would you like to review your follow-up appointment details?'\n"
    "   - 'Are you experiencing any of the warning signs mentioned in your discharge instructions?'\n"
    "4. **Route medical queries**: If the user asks a medical question (e.g., about symptoms, medications, treatments, side effects, "
    "dietary advice, or clinical guidance), you MUST return the string 'HANDOFF_TO_CLINICAL' to transfer control to the Clinical Agent.\n"
    "5. **Stay conversational**: Be friendly, empathetic, and helpful. Summarize key information from the report when relevant."
)

CLINICAL_SYSTEM_PROMPT = (
    f"You are the Clinical AI Agent specializing in Nephrology. Answer medical questions using the provided tools. {MEDICAL_DISCLAIMER}\n\n"
    "Workflow: 1. ALWAYS use the 'rag_query_tool' FIRST for medical questions to search the nephrology reference book. "
    "2. If the reference materials don't have sufficient information or the user asks about latest research, use the 'clinical_web_search' tool. "
    "3. When using RAG results, cite them as '[Source: Internal Nephrology Reference]'. "
    "4. When using web search, clearly state '[Source: General Internet Search]'. "
    "5. If the user asks a question that is NOT medical (e.g., 'Thank you', 'How are you?'), return the string 'HANDOFF_TO_RECEPTIONIST'."
)

# Create the agents based on API version
if USE_NEW_API:
    # Use new LangChain 1.0+ API
    receptionist_agent_graph = create_agent(LLM, RECEPTIONIST_TOOLS, system_prompt=RECEPTIONIST_SYSTEM_PROMPT)
    clinical_agent_graph = create_agent(LLM, CLINICAL_TOOLS, system_prompt=CLINICAL_SYSTEM_PROMPT)
    
    # Create wrapper functions to make them compatible with our node functions
    class AgentWrapper:
        def __init__(self, agent_graph):
            self.agent_graph = agent_graph
        
        def invoke(self, state):
            # Convert our state format to the format expected by the new agent API
            # The new API expects messages in the state
            try:
                # Get messages from state (could be a dict or the state itself)
                if isinstance(state, dict):
                    messages = state.get("messages", [])
                else:
                    messages = getattr(state, "messages", [])
                
                # Invoke the agent graph
                result = self.agent_graph.invoke({"messages": messages})
                
                # Extract the last message as output
                result_messages = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
                if result_messages:
                    last_msg = result_messages[-1]
                    if hasattr(last_msg, "content"):
                        output = last_msg.content
                    else:
                        output = str(last_msg)
                else:
                    output = ""
                return {"output": output, "messages": result_messages}
            except Exception as e:
                # Fallback error handling
                return {"output": f"Error: {str(e)}", "messages": state.get("messages", []) if isinstance(state, dict) else []}
    
    receptionist_agent = AgentWrapper(receptionist_agent_graph)
    clinical_agent = AgentWrapper(clinical_agent_graph)
else:
    # Use old API with prompts
    RECEPTIONIST_PROMPT = ChatPromptTemplate.from_messages([
        ("system", RECEPTIONIST_SYSTEM_PROMPT),
        ("placeholder", "{messages}")
    ])
    
    CLINICAL_PROMPT = ChatPromptTemplate.from_messages([
        ("system", CLINICAL_SYSTEM_PROMPT),
        ("placeholder", "{messages}")
    ])
    
    receptionist_agent_runnable = create_react_agent(LLM, RECEPTIONIST_TOOLS, RECEPTIONIST_PROMPT)
    clinical_agent_runnable = create_react_agent(LLM, CLINICAL_TOOLS, CLINICAL_PROMPT)
    
    # Wrap in executors for better error handling
    receptionist_agent = AgentExecutor(agent=receptionist_agent_runnable, tools=RECEPTIONIST_TOOLS, verbose=False)
    clinical_agent = AgentExecutor(agent=clinical_agent_runnable, tools=CLINICAL_TOOLS, verbose=False)

def receptionist_node(state: AgentState):
    SYSTEM_LOGGER.info("AGENT: Receptionist activated.")

    try:
        # Quick heuristic: if the last user message is clearly a medical question/symptom, force handoff
        user_query = ""
        if state.get("messages"):
            # find last human message content
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage) or (hasattr(msg, "content") and not isinstance(msg, AIMessage)):
                    user_query = msg.content if hasattr(msg, "content") else str(msg)
                    break

        # Medical keywords heuristic (keeps it simple but effective for routing)
        medical_keywords = [
            "swelling", "pain", "shortness of breath", "sob", "should i be worried",
            "should i", "symptom", "symptoms", "medication", "side effect", "side-effect",
            "diet", "dietary", "eating", "dizziness", "fever", "blood", "urine", "edema",
            "leg swelling", "leg swollen", "sore", "bleeding", "rash", "chest pain"
        ]
        if user_query:
            lowered = user_query.lower()
            if any(k in lowered for k in medical_keywords):
                # Short circuit: tell the receptionist to handoff to clinical agent
                SYSTEM_LOGGER.info("Heuristic: Detected medical question in Receptionist - routing to Clinical Agent.")
                ai_message = AIMessage(content="HANDOFF_TO_CLINICAL")
                return {"messages": [ai_message], "current_agent": "Clinical Agent", "patient_report": state.get("patient_report", "")}

        # Otherwise run the receptionist agent as usual (tool-based)
        result = receptionist_agent.invoke(state)

        # Extract the output message
        output_text = result.get("output", "")
        if isinstance(output_text, str):
            ai_message = AIMessage(content=output_text)
        else:
            ai_message = result.get("messages", [AIMessage(content=str(output_text))])[-1]

        # Extract patient report from tool results if available
        patient_report = state.get("patient_report", "")
        messages = result.get("messages", [])
        for msg in messages:
            if hasattr(msg, "content"):
                content = msg.content
                if "patient_name" in content and "discharge_date" in content:
                    try:
                        import json
                        if "{" in content and "}" in content:
                            start = content.find("{")
                            end = content.rfind("}") + 1
                            json_str = content[start:end]
                            patient_data = json.loads(json_str)
                            patient_report = json.dumps(patient_data, indent=2)
                            SYSTEM_LOGGER.info(f"PATIENT REPORT STORED: Retrieved report for {patient_data.get('patient_name', 'unknown')}")
                    except:
                        pass

        # If the model explicitly requested a handoff, perform it
        if isinstance(output_text, str) and "HANDOFF_TO_CLINICAL" in output_text:
            new_agent = "Clinical Agent"
            SYSTEM_LOGGER.info(f"HANDOFF: Receptionist -> {new_agent}")
            return {"messages": [ai_message], "current_agent": new_agent, "patient_report": patient_report}

        return {"messages": [ai_message], "current_agent": "Receptionist Agent", "patient_report": patient_report}
    except Exception as e:
        SYSTEM_LOGGER.exception(f"Error in receptionist_node: {e}")
        error_msg = AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")
        return {"messages": [error_msg], "current_agent": "Receptionist Agent", "patient_report": state.get("patient_report", "")}

def clinical_node(state: AgentState):
    SYSTEM_LOGGER.info("AGENT: Clinical Agent activated.")
    
    try:
        # Get the user's query for logging
        user_query = ""
        if state.get("messages"):
            last_user_msg = None
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage) or (hasattr(msg, "content") and not isinstance(msg, AIMessage)):
                    user_query = msg.content if hasattr(msg, "content") else str(msg)
                    break
        
        # The clinical agent will use its tools (RAG tool first, then web search if needed)
        # The agent executor handles the tool calling logic based on the prompt
        result = clinical_agent.invoke(state)
        
        # Extract the output message
        output_text = result.get("output", "")
        if isinstance(output_text, str):
            ai_message = AIMessage(content=output_text)
        else:
            ai_message = result.get("messages", [AIMessage(content=str(output_text))])[-1]
        
        # Log patient interaction with details (Requirement 5)
        SYSTEM_LOGGER.info(f"CLINICAL INTERACTION: User query='{user_query[:100]}' | Response generated | Length={len(output_text)} chars")
        
        # Check for explicit handoff instruction (routing)
        if "HANDOFF_TO_RECEPTIONIST" in output_text:
            new_agent = "Receptionist Agent"
            SYSTEM_LOGGER.info(f"HANDOFF: Clinical Agent -> {new_agent}")
            return {"messages": [ai_message], "current_agent": new_agent, "patient_report": state.get("patient_report", "")}
            
        return {"messages": [ai_message], "current_agent": "Clinical Agent", "patient_report": state.get("patient_report", "")}
    except Exception as e:
        SYSTEM_LOGGER.exception(f"Error in clinical_node: {e}")
        error_msg = AIMessage(content=f"I encountered an error while processing your medical question: {str(e)}. Please try again or rephrase your question.")
        return {"messages": [error_msg], "current_agent": "Clinical Agent", "patient_report": state.get("patient_report", "")}

def route_agent(state: AgentState) -> str:
    """Routes the flow based on which agent has control. Returns node name or END."""
    current_agent = state.get('current_agent', 'Receptionist Agent')
    messages = state.get("messages", [])
    
    # Check if we have a response from an agent (prevent infinite loops)
    if len(messages) >= 2:
        last_msg = messages[-1]
        # Check if last message is an AI response
        if isinstance(last_msg, AIMessage):
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            
            # Check for handoff signals - these require continuing
            if "HANDOFF_TO_CLINICAL" in content and current_agent == "Receptionist Agent":
                return "clinical_agent_node"
            elif "HANDOFF_TO_RECEPTIONIST" in content and current_agent == "Clinical Agent":
                return "receptionist_agent_node"
            
            # If we have a substantial response without handoff, end the graph
            # This prevents infinite loops when agent has already responded
            if len(content) > 15 and "HANDOFF" not in content:
                return END
    
    # Route based on current agent (for initial routing or handoffs)
    if current_agent == "Clinical Agent":
        return "clinical_agent_node"
    elif current_agent == "Receptionist Agent":
        return "receptionist_agent_node"
    else:
        return END  # Default to end if unknown


# 1. Create the graph
workflow = StateGraph(AgentState)

# 2. Add the nodes
workflow.add_node("receptionist_agent_node", receptionist_node)
workflow.add_node("clinical_agent_node", clinical_node)

# 3. Set the entry point (must start with the Receptionist)
workflow.set_entry_point("receptionist_agent_node")

# 4. Define the edges (transitions)
# Use conditional edges to route based on agent state
# Add END condition to prevent infinite loops

workflow.add_conditional_edges(
    "receptionist_agent_node",
    route_agent,
    {
        "clinical_agent_node": "clinical_agent_node",
        "receptionist_agent_node": "receptionist_agent_node",
        END: END
    }
)

workflow.add_conditional_edges(
    "clinical_agent_node",
    route_agent,
    {
        "receptionist_agent_node": "receptionist_agent_node",
        "clinical_agent_node": "clinical_agent_node",
        END: END
    }
)

# 5. Compile the graph with recursion limit configuration
app = workflow.compile()
# --- Compatibility wrapper to invoke the compiled graph from Streamlit ---
def invoke_app(state: AgentState):
    """
    Invokes the compiled graph with proper configuration to prevent infinite loops.
    Returns a state-like dict expected by Streamlit.
    """
    try:
        # Use invoke with recursion limit to prevent infinite loops
        SYSTEM_LOGGER.info("Attempting app.invoke(...) with recursion limit")
        # Invoke with config to set recursion limit
        config = {"recursion_limit": 10}  # Limit to 10 iterations max
        result = app.invoke(state, config=config)
        return result
    except Exception as e_invoke:
        SYSTEM_LOGGER.exception(f"app.invoke failed: {e_invoke}")
        # Return the state as-is so the app doesn't crash
        return state
