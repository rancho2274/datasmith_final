import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from logging_setup import SYSTEM_LOGGER  # reuse central logger
db_logger = SYSTEM_LOGGER.getChild("database_access")

# 1. Load the dummy patient data
try:
    with open("patient_data.json", "r") as f:
        PATIENT_DATABASE = json.load(f)
except FileNotFoundError:
    print("FATAL: patient_data.json not found. Run Step 1.1 script first.")
    PATIENT_DATABASE = {}

# 2. Define the input schema for the tool using Pydantic
class PatientLookupInput(BaseModel):
    """Input for looking up a patient's discharge report."""
    patient_name: str = Field(description="The full name of the patient (e.g., 'John Smith').")

# 3. Define the dedicated tool for the Receptionist Agent
@tool(args_schema=PatientLookupInput)
def get_patient_discharge_report(patient_name: str) -> str:
    """
    Retrieves the complete post-discharge report for a patient given their full name.
    
    Returns a JSON string of the report or an error message if not found.
    Handles error cases (patient not found, multiple matches). 
    """
    
    db_logger.info(f"Attempting to retrieve report for: {patient_name}")
    
    # Handle case-insensitive search and partial matches
    patient_name_lower = patient_name.lower().strip()
    matches = []
    
    for db_name in PATIENT_DATABASE.keys():
        if db_name.lower() == patient_name_lower:
            matches.append(db_name)
        elif patient_name_lower in db_name.lower() or db_name.lower() in patient_name_lower:
            matches.append(db_name)
    
    # Handle multiple matches
    if len(matches) > 1:
        db_logger.warning(f"ERROR: Multiple patients found matching '{patient_name}': {matches}")
        return f"ERROR: Multiple patients found with similar names: {', '.join(matches)}. Please provide the full exact name."
    
    # Handle single match
    if len(matches) == 1:
        exact_name = matches[0]
        report = PATIENT_DATABASE[exact_name]
        db_logger.info(f"SUCCESS: Report retrieved for {exact_name}.")
        return json.dumps({"patient_name": exact_name, **report}, indent=2)
    
    # Handle no matches
    db_logger.warning(f"ERROR: Patient not found: {patient_name}")
    return f"ERROR: Patient '{patient_name}' not found in the database. Please check the spelling or confirm the patient's identity."


# Web Search Tool Setup (Requirement 4)
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Define the input schema for the web search tool
class WebSearchInput(BaseModel):
    """Input for performing a web search."""
    query: str = Field(description="The search query to execute (e.g., 'latest research on SGLT2 inhibitors for kidney disease').")

# 2. Initialize the DuckDuckGo Search wrapper
# This provides the base search functionality
ddg_search = DuckDuckGoSearchRun()

@tool(args_schema=WebSearchInput)
def clinical_web_search(query: str) -> str:
    """
    Performs a real-time web search for information outside the internal
    nephrology reference book, specifically for latest research or external
    clinical data.
    """
    
    # Use the same logger for web search attempts (Requirement 5.2)
    db_logger.info(f"WEB SEARCH: Attempting search for query: '{query}'")
    
    try:
        # Run the search, limiting results to keep the response concise
        # The .run() method returns a formatted string of search results (title, snippet, source)
        results = ddg_search.run(query)
        
        # Format the results with a clear indication of the source
        response = f"WEB SEARCH RESULTS (Source: General Internet Search):\n{results}"
        db_logger.info("WEB SEARCH SUCCESS: Results obtained.")
        return response
    
    except Exception as e:
        db_logger.error(f"WEB SEARCH ERROR: Failed to execute search. Error: {e}")
        return "WEB SEARCH ERROR: I was unable to perform the external search. Please try rephrasing your query."

# Example of tools list for the Clinical Agent:
# CLINICAL_TOOLS = [rag_chain_tool, clinical_web_search]