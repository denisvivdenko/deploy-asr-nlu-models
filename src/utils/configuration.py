import logging
import os
from datetime import datetime
import yaml


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


def load_params(params_fpath: str) -> dict:
    """
    params_fpath: str (file path to yaml file)
    return: dict
    """
    with open(params_fpath, "r") as file:
        return yaml.safe_load(file)
