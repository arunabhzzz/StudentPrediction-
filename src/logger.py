import logging
import os
from datetime import datetime

# Step 1: Create a log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Step 2: Join current working directory with "logs" folder and log file name
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)  # Create the 'logs' folder if it doesn't exist

# Step 3: Full path to the log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Step 4: Configure logging to write to the file
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
