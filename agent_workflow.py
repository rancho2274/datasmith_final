from typing import TypedDict, Annotated, List
import operator
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# ---- Agent creation imports ----
USE_NEW_API = False
try:
    from langchain.agents import create_agent
    USE_NEW_API = True
except ImportError:
    try:
        from langchain.agents import create_react_agent, AgentExecutor
    except ImportError:
        try:
            from langchain.agents.react.agent import create_react_agent
            from langchain.agents import AgentExecutor
        except ImportError:
            from langchain.agents import AgentExecutor, create_openai_tools_agent
            from langchain_core.prompts import MessagesPlaceholder
            def create_react_agent(llm, tools, prompt):
                return create_openai_tools_agent(llm, tools, prompt)

from logging_setup import SYSTEM_LOGGER
from tools import get_patient_discharge_report, clinical_web_search
from rag_setup import RAG_RETRIEVAL_CHAIN
from langchain_core.tools import tool

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_agent: str
    patient_report: str

LLM = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

@tool
def rag_query_tool(query: str) -> str:
    if RAG_RETRIEVAL_CHAIN is None:
        return "ERROR: RAG system not initialized."
    try:
        result = RAG_RETRIEVAL_CHAIN.invoke({"input": query})
        rag_output = result.get("output", "No relevant information found.")
        return rag_output
    except Exception as e:
        return f"ERROR: Unable to query reference materials. Error: {e}"

RECEPTIONIST_TOOLS = [get_patient_discharge_report]
CLINICAL_TOOLS = [rag_query_tool, clinical_web_search]

MEDICAL_DISCLAIMER = (
    "NOTE: This is an AI assistant for educational purposes only. "
    "Always consult healthcare professionals for medical advice."
)

RECEPTIONIST_SYSTEM_PROMPT = (
    f"You are the Post-Discharge Receptionist AI. {MEDICAL_DISCLAIMER}\n"
    "1. Greet the patient and ask for their full name.\n"
    "2. Retrieve the report using the tool once.\n"
    "3. After retrieving, ask follow-up questions.\n"
    "4. For medical questions, return 'HANDOFF_TO_CLINICAL'.\n"
    "5. Be empathetic and helpful."
)

CLINICAL_SYSTEM_PROMPT = (
    f"You are the Clinical AI Agent specializing in Nephrology. {MEDICAL_DISCLAIMER}\n"
    "1. ALWAYS use rag_query_tool first.\n"
    "2. If insufficient data, use clinical_web_search.\n"
    "3. RAG results → cite [Source: Internal Nephrology Reference].\n"
    "4. Web search → cite [Source: General Internet Search].\n"
    "5. If non-medical: return 'HANDOFF_TO_RECEPTIONIST'."
)

if USE_NEW_API:
    receptionist_agent_graph = create_agent(LLM, RECEPTIONIST_TOOLS, system_prompt=RECEPTIONIST_SYSTEM_PROMPT)
    clinical_agent_graph = create_agent(LLM, CLINICAL_TOOLS, system_prompt=CLINICAL_SYSTEM_PROMPT)

    class AgentWrapper:
        def __init__(self, agent_graph):
            self.agent_graph = agent_graph
        
        def invoke(self, state):
            try:
                messages = state.get("messages", [])
                result = self.agent_graph.invoke({"messages": messages})
                result_messages = result.get("messages", [])
                if result_messages:
                    content = getattr(result_messages[-1], "content", str(result_messages[-1]))
                else:
                    content = ""
                return {"output": content, "messages": result_messages}
            except Exception as e:
                return {"output": f"Error: {str(e)}", "messages": state.get("messages", [])}

    receptionist_agent = AgentWrapper(receptionist_agent_graph)
    clinical_agent = AgentWrapper(clinical_agent_graph)

else:
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

    receptionist_agent = AgentExecutor(agent=receptionist_agent_runnable, tools=RECEPTIONIST_TOOLS, verbose=False)
    clinical_agent = AgentExecutor(agent=clinical_agent_runnable, tools=CLINICAL_TOOLS, verbose=False)

def receptionist_node(state: AgentState):
    try:
        user_query = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_query = msg.content.lower()
                break

        medical_keywords = [
            "swelling", "pain", "shortness of breath", "symptom", "medication",
            "side effect", "diet", "dizziness", "fever", "blood", "urine", "rash"
        ]

        if any(k in user_query for k in medical_keywords):
            ai_msg = AIMessage(content="HANDOFF_TO_CLINICAL")
            return {"messages": [ai_msg], "current_agent": "Clinical Agent", "patient_report": state.get("patient_report", "")}

        result = receptionist_agent.invoke(state)
        output_text = result.get("output", "")
        ai_message = AIMessage(content=output_text)

        patient_report = state.get("patient_report", "")
        for msg in result.get("messages", []):
            content = getattr(msg, "content", "")
            if "patient_name" in content and "discharge_date" in content:
                import json
                try:
                    data = json.loads(content[content.find("{"): content.rfind("}") + 1])
                    patient_report = json.dumps(data, indent=2)
                except:
                    pass

        if "HANDOFF_TO_CLINICAL" in output_text:
            return {"messages": [ai_message], "current_agent": "Clinical Agent", "patient_report": patient_report}

        return {"messages": [ai_message], "current_agent": "Receptionist Agent", "patient_report": patient_report}

    except Exception as e:
        error_msg = AIMessage(content=f"Error: {str(e)}")
        return {"messages": [error_msg], "current_agent": "Receptionist Agent", "patient_report": state.get("patient_report", "")}

def clinical_node(state: AgentState):
    try:
        result = clinical_agent.invoke(state)
        output_text = result.get("output", "")
        ai_message = AIMessage(content=output_text)

        if "HANDOFF_TO_RECEPTIONIST" in output_text:
            return {"messages": [ai_message], "current_agent": "Receptionist Agent", "patient_report": state.get("patient_report", "")}

        return {"messages": [ai_message], "current_agent": "Clinical Agent", "patient_report": state.get("patient_report", "")}

    except Exception as e:
        error_msg = AIMessage(content=f"Medical processing error: {str(e)}")
        return {"messages": [error_msg], "current_agent": "Clinical Agent", "patient_report": state.get("patient_report", "")}

def route_agent(state: AgentState) -> str:
    current_agent = state.get("current_agent", "Receptionist Agent")
    messages = state.get("messages", [])

    if len(messages) >= 2 and isinstance(messages[-1], AIMessage):
        content = messages[-1].content
        if "HANDOFF_TO_CLINICAL" in content:
            return "clinical_agent_node"
        if "HANDOFF_TO_RECEPTIONIST" in content:
            return "receptionist_agent_node"
        if len(content) > 15:
            return END

    return "clinical_agent_node" if current_agent == "Clinical Agent" else "receptionist_agent_node"

workflow = StateGraph(AgentState)
workflow.add_node("receptionist_agent_node", receptionist_node)
workflow.add_node("clinical_agent_node", clinical_node)
workflow.set_entry_point("receptionist_agent_node")

workflow.add_conditional_edges(
    "receptionist_agent_node",
    route_agent,
    {"clinical_agent_node": "clinical_agent_node", "receptionist_agent_node": "receptionist_agent_node", END: END}
)

workflow.add_conditional_edges(
    "clinical_agent_node",
    route_agent,
    {"receptionist_agent_node": "receptionist_agent_node", "clinical_agent_node": "clinical_agent_node", END: END}
)

app = workflow.compile()

def invoke_app(state: AgentState):
    try:
        return app.invoke(state, config={"recursion_limit": 10})
    except Exception:
        return state
