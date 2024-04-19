import sys
from src.exception import CustomException
from src.logger import logging
import os
import dill
import pickle


def store_data(data_store_path, data):   
    try:
        with open(os.makedirs(data_store_path, exist_ok=True), 'wb') as file:
            file.write(data)
            file.close()
        logging.info("Raw data file stored.")

    except Exception  as e:
        raise CustomException(e, sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)