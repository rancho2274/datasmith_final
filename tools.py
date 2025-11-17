import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from logging_setup import SYSTEM_LOGGER
db_logger = SYSTEM_LOGGER.getChild("database_access")

try:
    with open("patient_data.json", "r") as f:
        PATIENT_DATABASE = json.load(f)
except FileNotFoundError:
    print("FATAL: patient_data.json not found. Run Step 1.1 script first.")
    PATIENT_DATABASE = {}

class PatientLookupInput(BaseModel):
    patient_name: str = Field(description="The full name of the patient (e.g., 'John Smith').")

@tool(args_schema=PatientLookupInput)
def get_patient_discharge_report(patient_name: str) -> str:
    """
    Retrieves the complete post-discharge report for a patient given their full name.
    
    Returns a JSON string of the report or an error message if not found.
    Handles error cases (patient not found, multiple matches). 
    """
    
    db_logger.info(f"Attempting to retrieve report for: {patient_name}")
    
    patient_name_lower = patient_name.lower().strip()
    matches = []
    
    for db_name in PATIENT_DATABASE.keys():
        if db_name.lower() == patient_name_lower:
            matches.append(db_name)
        elif patient_name_lower in db_name.lower() or db_name.lower() in patient_name_lower:
            matches.append(db_name)
    
    if len(matches) > 1:
        db_logger.warning(f"ERROR: Multiple patients found matching '{patient_name}': {matches}")
        return f"ERROR: Multiple patients found with similar names: {', '.join(matches)}. Please provide the full exact name."
    
    if len(matches) == 1:
        exact_name = matches[0]
        report = PATIENT_DATABASE[exact_name]
        db_logger.info(f"SUCCESS: Report retrieved for {exact_name}.")
        return json.dumps({"patient_name": exact_name, **report}, indent=2)
    
    db_logger.warning(f"ERROR: Patient not found: {patient_name}")
    return f"ERROR: Patient '{patient_name}' not found in the database. Please check the spelling or confirm the patient's identity."

from langchain_community.tools import DuckDuckGoSearchRun

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to execute (e.g., 'latest research on SGLT2 inhibitors for kidney disease').")

ddg_search = DuckDuckGoSearchRun()

@tool(args_schema=WebSearchInput)
def clinical_web_search(query: str) -> str:
    """
    Performs a real-time web search for information outside the internal
    nephrology reference book, specifically for latest research or external
    clinical data.
    """
    
    db_logger.info(f"WEB SEARCH: Attempting search for query: '{query}'")
    
    try:
        results = ddg_search.run(query)
        response = f"WEB SEARCH RESULTS (Source: General Internet Search):\n{results}"
        db_logger.info("WEB SEARCH SUCCESS: Results obtained.")
        return response
    
    except Exception as e:
        db_logger.error(f"WEB SEARCH ERROR: Failed to execute search. Error: {e}")
        return "WEB SEARCH ERROR: I was unable to perform the external search. Please try rephrasing your query."

# CLINICAL_TOOLS = [rag_chain_tool, clinical_web_search]