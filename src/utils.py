import sys
from src.exception import CustomException
from src.logger import logging
import os

def store_data(data_store_path, data):   
    try:
        with open(os.makedirs(data_store_path, exist_ok=True), 'wb') as file:
            file.write(data)
            file.close()
        logging.info("Raw data file stored.")

    except Exception  as e:
        raise CustomException(e, sys)