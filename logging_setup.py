import logging
import os

LOG_FILE = "system_logs.log"

if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
    except (PermissionError, OSError):
        try:
            with open(LOG_FILE, 'w') as f:
                f.write('')
        except (PermissionError, OSError):
            pass

SYSTEM_LOGGER = logging.getLogger('SystemFlow')
SYSTEM_LOGGER.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    file_handler.setFormatter(formatter)
    SYSTEM_LOGGER.addHandler(file_handler)
except (PermissionError, OSError) as e:
    print(f"Warning: Could not create log file handler: {e}. Using console logging only.")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
SYSTEM_LOGGER.addHandler(console_handler)

SYSTEM_LOGGER.info("SYSTEM INITIALIZED: Comprehensive logging started.")