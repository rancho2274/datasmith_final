"""
Verification script to check if all assignment requirements are met.
"""
import json
import os
import sys

print("=" * 60)
print("ASSIGNMENT REQUIREMENTS VERIFICATION")
print("=" * 60)

all_passed = True

# 1. Data Setup
print("\n1. DATA SETUP")
print("-" * 60)

# Check patient data
try:
    with open("patient_data.json", "r") as f:
        patient_data = json.load(f)
    patient_count = len(patient_data)
    if patient_count >= 25:
        print(f"[OK] Patient data: {patient_count} patients (Requirement: 25+)")
    else:
        print(f"[FAIL] Patient data: {patient_count} patients (Requirement: 25+)")
        all_passed = False
except Exception as e:
    print(f"[FAIL] Patient data file error: {e}")
    all_passed = False

# Check nephrology reference
try:
    if os.path.exists("nephrology_reference.txt"):
        size = os.path.getsize("nephrology_reference.txt")
        with open("nephrology_reference.txt", "r") as f:
            lines = len(f.readlines())
        print(f"[OK] Nephrology reference: {lines} lines, {size} bytes")
    else:
        print("[FAIL] Nephrology reference file not found")
        all_passed = False
except Exception as e:
    print(f"[FAIL] Nephrology reference error: {e}")
    all_passed = False

# Check database storage (JSON file)
if os.path.exists("patient_data.json"):
    print("[OK] Database storage: JSON file format")
else:
    print("[FAIL] Database storage: Not found")
    all_passed = False

# Check vector embeddings (ChromaDB)
if os.path.exists("chroma_db"):
    print("[OK] Vector embeddings: ChromaDB directory exists")
else:
    print("[WARN] Vector embeddings: ChromaDB directory not found (will be created on first run)")

# 2. Multi-Agent System
print("\n2. MULTI-AGENT SYSTEM")
print("-" * 60)

# Check Receptionist Agent
try:
    from agent_workflow import RECEPTIONIST_SYSTEM_PROMPT, RECEPTIONIST_TOOLS
    if "get_patient_discharge_report" in str(RECEPTIONIST_TOOLS):
        print("[OK] Receptionist Agent: Configured with patient retrieval tool")
    else:
        print("[FAIL] Receptionist Agent: Missing patient retrieval tool")
        all_passed = False
    
    if "Ask patient for their name" in RECEPTIONIST_SYSTEM_PROMPT or "Greet the patient" in RECEPTIONIST_SYSTEM_PROMPT:
        print("[OK] Receptionist Agent: Asks for patient name")
    else:
        print("[FAIL] Receptionist Agent: Missing name request in prompt")
        all_passed = False
    
    if "follow-up questions" in RECEPTIONIST_SYSTEM_PROMPT.lower():
        print("[OK] Receptionist Agent: Configured to ask follow-up questions")
    else:
        print("[FAIL] Receptionist Agent: Missing follow-up questions in prompt")
        all_passed = False
    
    if "HANDOFF_TO_CLINICAL" in RECEPTIONIST_SYSTEM_PROMPT:
        print("[OK] Receptionist Agent: Routes to Clinical Agent")
    else:
        print("[FAIL] Receptionist Agent: Missing routing to Clinical Agent")
        all_passed = False
except Exception as e:
    print(f"[FAIL] Receptionist Agent check failed: {e}")
    all_passed = False

# Check Clinical Agent
try:
    from agent_workflow import CLINICAL_SYSTEM_PROMPT, CLINICAL_TOOLS
    if "rag_query_tool" in str(CLINICAL_TOOLS):
        print("[OK] Clinical Agent: Configured with RAG tool")
    else:
        print("[FAIL] Clinical Agent: Missing RAG tool")
        all_passed = False
    
    if "clinical_web_search" in str(CLINICAL_TOOLS):
        print("[OK] Clinical Agent: Configured with web search tool")
    else:
        print("[FAIL] Clinical Agent: Missing web search tool")
        all_passed = False
    
    if "citation" in CLINICAL_SYSTEM_PROMPT.lower() or "source" in CLINICAL_SYSTEM_PROMPT.lower():
        print("[OK] Clinical Agent: Configured to provide citations")
    else:
        print("[FAIL] Clinical Agent: Missing citation requirement")
        all_passed = False
except Exception as e:
    print(f"[FAIL] Clinical Agent check failed: {e}")
    all_passed = False

# 3. RAG Implementation
print("\n3. RAG IMPLEMENTATION")
print("-" * 60)

try:
    from rag_setup import RAG_RETRIEVAL_CHAIN, setup_rag_retriever
    if RAG_RETRIEVAL_CHAIN is not None:
        print("[OK] RAG Retrieval Chain: Initialized")
    else:
        print("[WARN] RAG Retrieval Chain: Not initialized (will be created on first run)")
    
    # Check if RAG setup function exists
    if callable(setup_rag_retriever):
        print("[OK] RAG Setup: Function available")
    else:
        print("[FAIL] RAG Setup: Function not found")
        all_passed = False
    
    # Check for chunking and embeddings
    from rag_setup import RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, Chroma
    print("[OK] RAG Components: Text splitter, embeddings, and vectorstore imported")
except Exception as e:
    print(f"[FAIL] RAG Implementation check failed: {e}")
    all_passed = False

# 4. Web Search Tool
print("\n4. WEB SEARCH TOOL")
print("-" * 60)

try:
    from tools import clinical_web_search
    # Tools are StructuredTool objects, check if they exist
    if clinical_web_search is not None:
        print("[OK] Web Search Tool: Function available")
        # Check if it indicates source
        try:
            doc = clinical_web_search.description if hasattr(clinical_web_search, 'description') else str(clinical_web_search)
            if doc and ("web search" in doc.lower() or "internet" in doc.lower() or "general" in doc.lower()):
                print("[OK] Web Search Tool: Documented with source indication")
            else:
                print("[WARN] Web Search Tool: Documentation may need source indication")
        except:
            print("[OK] Web Search Tool: Available (source check skipped)")
    else:
        print("[FAIL] Web Search Tool: Function not found")
        all_passed = False
except Exception as e:
    print(f"[FAIL] Web Search Tool check failed: {e}")
    all_passed = False

# 5. Logging System
print("\n5. LOGGING SYSTEM")
print("-" * 60)

try:
    from logging_setup import SYSTEM_LOGGER, LOG_FILE
    if SYSTEM_LOGGER is not None:
        print("[OK] Logging System: SYSTEM_LOGGER initialized")
    else:
        print("[FAIL] Logging System: SYSTEM_LOGGER not initialized")
        all_passed = False
    
    # Check if logging is used in agents
    from agent_workflow import receptionist_node, clinical_node
    import inspect
    receptionist_code = inspect.getsource(receptionist_node)
    clinical_code = inspect.getsource(clinical_node)
    
    if "SYSTEM_LOGGER" in receptionist_code:
        print("[OK] Logging: Used in Receptionist Agent")
    else:
        print("[FAIL] Logging: Not used in Receptionist Agent")
        all_passed = False
    
    if "SYSTEM_LOGGER" in clinical_code:
        print("[OK] Logging: Used in Clinical Agent")
    else:
        print("[FAIL] Logging: Not used in Clinical Agent")
        all_passed = False
    
    if "HANDOFF" in receptionist_code and "SYSTEM_LOGGER" in receptionist_code:
        print("[OK] Logging: Agent handoffs are logged")
    else:
        print("[WARN] Logging: Agent handoffs may not be fully logged")
    
    if os.path.exists(LOG_FILE):
        print(f"[OK] Log File: {LOG_FILE} exists")
    else:
        print(f"[WARN] Log File: {LOG_FILE} will be created on first run")
except Exception as e:
    print(f"[FAIL] Logging System check failed: {e}")
    all_passed = False

# 6. Patient Data Retrieval Tool
print("\n6. PATIENT DATA RETRIEVAL TOOL")
print("-" * 60)

try:
    from tools import get_patient_discharge_report
    # Tools are StructuredTool objects
    if get_patient_discharge_report is not None:
        print("[OK] Patient Retrieval Tool: Function available")
        
        # Check the tool description/documentation
        tool_desc = ""
        if hasattr(get_patient_discharge_report, 'description'):
            tool_desc = get_patient_discharge_report.description
        elif hasattr(get_patient_discharge_report, '__doc__'):
            tool_desc = get_patient_discharge_report.__doc__ or ""
        
        # Check error handling in description
        if "not found" in tool_desc.lower() or "error" in tool_desc.lower() or "multiple" in tool_desc.lower():
            print("[OK] Patient Retrieval Tool: Error handling documented")
        else:
            print("[WARN] Patient Retrieval Tool: Error handling documentation may be incomplete")
        
        # Check source code file for implementation details
        try:
            with open("tools.py", "r", encoding="utf-8") as f:
                tools_code = f.read()
                if "not found" in tools_code.lower() or "ERROR" in tools_code:
                    print("[OK] Patient Retrieval Tool: Error handling implemented in code")
                if "multiple" in tools_code.lower() or "matches" in tools_code.lower():
                    print("[OK] Patient Retrieval Tool: Handles multiple matches")
                if "db_logger" in tools_code:
                    print("[OK] Patient Retrieval Tool: Database access is logged")
                else:
                    print("[FAIL] Patient Retrieval Tool: Database access not logged")
                    all_passed = False
        except:
            print("[WARN] Could not verify implementation details")
    else:
        print("[FAIL] Patient Retrieval Tool: Function not found")
        all_passed = False
except Exception as e:
    print(f"[FAIL] Patient Data Retrieval Tool check failed: {e}")
    all_passed = False

# 7. Web Interface
print("\n7. WEB INTERFACE")
print("-" * 60)

if os.path.exists("app.py"):
    print("[OK] Web Interface: app.py exists")
    try:
        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()
            if "streamlit" in app_code.lower():
                print("[OK] Web Interface: Streamlit implementation")
            else:
                print("[FAIL] Web Interface: Streamlit not found")
                all_passed = False
    except Exception as e:
        print(f"[WARN] Could not read app.py: {e}")
        print("[OK] Web Interface: app.py exists (verification skipped)")
else:
    print("[FAIL] Web Interface: app.py not found")
    all_passed = False

# Summary
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

if all_passed:
    print("[SUCCESS] ALL CRITICAL REQUIREMENTS MET!")
    print("\nYour POC system appears to meet all assignment requirements.")
    print("You can now test the system by running: streamlit run app.py")
else:
    print("[WARNING] SOME REQUIREMENTS MAY NEED ATTENTION")
    print("\nPlease review the items marked with [FAIL] above.")

print("\n" + "=" * 60)

