import logging
from datetime import datetime
import os
LOG_DIR="energy_logs"
os.chdir(r"E:\Ineuron\Project\energy_efficiency\energy-ML_project")

CURRENT_TIME_STAMP=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

LOG_FILE_NAME=f"log_{CURRENT_TIME_STAMP}.log"

os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH=os.path.join(LOG_DIR,LOG_FILE_NAME)


logging.basicConfig(filename=LOG_FILE_PATH,
filemode="w",
format='[%(asctime)s] - %(created)f - %(levelname)s - %(lineno)d - %(message)s',
level=logging.INFO)