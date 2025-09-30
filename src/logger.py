import logging 
import os 
from datetime import datetime

path=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"  # create file name
log_dir=os.path.join(os.getcwd() , 'logs')                  # find current workng dir and name of folder to be create logs
os.makedirs(log_dir , exist_ok=True)                        # create folder 
log_file_path=os.path.join(log_dir , path)                  # overall file path

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="[%(asctime)s]- %(name)s- %(levelname)s -%(message)s"
)
