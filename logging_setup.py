import logging
import os

# --- Central Logging Configuration (Requirement 5) ---
LOG_FILE = "system_logs.log"

# Clear old log file for a fresh run (handle file lock gracefully)
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
    except (PermissionError, OSError):
        # File is locked by another process (e.g., previous Streamlit instance)
        # Just truncate it instead or append to it
        try:
            # Try to truncate the file instead
            with open(LOG_FILE, 'w') as f:
                f.write('')  # Clear contents
        except (PermissionError, OSError):
            # If we still can't access it, just continue - logs will append
            pass

SYSTEM_LOGGER = logging.getLogger('SystemFlow')
SYSTEM_LOGGER.setLevel(logging.INFO)

# Formatter includes timestamp and a clear message structure
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File Handler: Logs all system activity (handle file lock gracefully)
try:
    file_handler = logging.FileHandler(LOG_FILE, mode='a')  # Use append mode to avoid conflicts
    file_handler.setFormatter(formatter)
    SYSTEM_LOGGER.addHandler(file_handler)
except (PermissionError, OSError) as e:
    # If we can't create a file handler, just use console logging
    print(f"Warning: Could not create log file handler: {e}. Using console logging only.")

# Console Handler: For real-time visibility during development
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
SYSTEM_LOGGER.addHandler(console_handler)

SYSTEM_LOGGER.info("SYSTEM INITIALIZED: Comprehensive logging started.")

# You can now use SYSTEM_LOGGER.info(), .warning(), or .error() throughout your code.
# The tool loggers (like db_logger from tools.py) should also be set up to feed into this, 
# or you can simply use SYSTEM_LOGGER directly in the agent logic.