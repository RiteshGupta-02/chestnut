import logging
import os
from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
# logs_path = os.path.join(os.getcwd(),"logs")
# os.makedirs(logs_path,exist_ok=True)

# LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     format = '[%(asctime)s] %(lineno)d - %(levelname)s - %(message)s',
#     level = logging.INFO,
#     force = True
# )

logger = logging.getLogger(__name__)

def setup_logger():
    if logger.hasHandlers():
        return logger

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = datetime.now().strftime("%d_%m_%Y_%H_%M_%S.log")

    logging.basicConfig(
        filename=os.path.join(log_dir, log_file),
        format='[%(asctime)s] %(lineno)d - %(levelname)s - %(message)s',
        level=logging.INFO,
        force=True
    )

    return logger
logger.info("Logging has started")
if __name__ == "__main__":
    logger.info("Logging has started")