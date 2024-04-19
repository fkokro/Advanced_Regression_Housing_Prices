import pandas as pd
import requests
import os
from dataclasses import dataclass
import sys
from src.exception import CustomException
from src.logger import logging
import psycopg2
import json
from configuration import local_settings

# Load config file
config_postgres = local_settings.config_postgres

@dataclass
class DataIngestionConfig:
    raw_data_json_path: str=os.path.join('artifacts','raw_data.json')
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    train_data_txt_path: str=os.path.join('artifacts','train.txt')
    test_data_txt_path: str=os.path.join('artifacts','test.txt')
    raw_data_path: str=os.path.join('artifacts','data.csv')   
    

class DataETL:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
        
    def api_extract(self, API_URL) -> dict:
        """Extract data from an API source. Store and return"""
        try:
            self.data = requests.get(API_URL)
            logging.info('Data successful extracted from {}'.format(API_URL))
            return self.data

        except Exception as e:
            raise CustomException(e, sys)
        
    def csv_extract(self, filename):
        """Import csv files"""
        try:
            df = pd.read_csv(filename)
            logging.info("{} imported".format(filename))
            return df
        except Exception as e:
            raise CustomException(e, sys)
    
    def fill_none_housing_data(self, df):
        """Fillna values for the housing data set, both categorical and numeric features."""
        for col in df.columns:
            check = df[col].dtypes
            if check == 'object':
                df.fillna({col:'None'}, inplace=True)
            else:
                df.fillna({col:df[col].mean()}, inplace=True)
        
        return df
        logging.info('Null values filled.')

    def process_housing_file(self):
        """Must enter data path from DataIngestionConfig"""
        try:
            train_df = self.csv_extract('notebooks/data/train.csv')
            test_df = self.csv_extract('notebooks/data/test.csv')
            train_df.rename(columns={'1stFlrSF':'FirstFlrSF','2ndFlrSF':'SeondFlrSF','3SsnPorch':'ThreeSsnPorch'}, inplace=True)
            test_df.rename(columns={'1stFlrSF':'FirstFlrSF','2ndFlrSF':'SeondFlrSF','3SsnPorch':'ThreeSsnPorch'}, inplace=True)
            trained_processed_df = self.fill_none_housing_data(train_df)
            test_processed_df = self.fill_none_housing_data(test_df)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_txt_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_txt_path), exist_ok=True)
            trained_processed_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            trained_processed_df.to_csv(self.ingestion_config.train_data_txt_path, sep='|', index=False)
            test_processed_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            test_processed_df.to_csv(self.ingestion_config.test_data_txt_path, sep='|', index=False)
            logging.info('Data processed and stored in artifacts.')
            
        except Exception as e:
            raise CustomException(e, sys)

    def create_train_table_dev(self):
        schema = config_postgres['schema']
        try:
            conn = psycopg2.connect(
                host = config_postgres['hostname'],
                dbname = config_postgres['database'],
                user = config_postgres['username'],
                password = config_postgres['pwd'],
                port = config_postgres['port_id']
            )
            
            cur = conn.cursor()
            
            with open('sql/create_processed_train_table.sql','r') as f:
                create_processed_train_table_script = f.read()
                f.close()
            cur.execute(create_processed_train_table_script)
            conn.commit()

        except Exception as e:
              raise CustomException(e, sys)
        
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()
          
    def create_test_table_dev(self):
        schema = config_postgres['schema']
        try:
            conn = psycopg2.connect(
                host = config_postgres['hostname'],
                dbname = config_postgres['database'],
                user = config_postgres['username'],
                password = config_postgres['pwd'],
                port = config_postgres['port_id']
            )
            
            cur = conn.cursor()
            
            with open('sql/create_processed_test_table.sql','r') as f:
                create_processed_test_table_script = f.read()
                f.close()
            cur.execute(create_processed_test_table_script)
            conn.commit()

        except Exception as e:
              raise CustomException(e, sys)
  
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()
                

    def load_train_table_dev(self):
        schema = config_postgres['schema']
        try:
            conn = psycopg2.connect(
                host = config_postgres['hostname'],
                dbname = config_postgres['database'],
                user = config_postgres['username'],
                password = config_postgres['pwd'],
                port = config_postgres['port_id']
            )
            
            cur = conn.cursor()
            query = """COPY ML_DEV.HOUSING_PROCESSED_TRAIN_DATA FROM STDIN WITH DELIMITER as '|'"""
            with open('artifacts/train.txt','r') as file:
                next(file)
                cur.copy_expert(
                    query,
                    file
                )
                conn.commit()
                file.close()

        except Exception as e:
              raise CustomException(e, sys)

        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()
          
    def load_test_table_dev(self):
        schema = config_postgres['schema']
        try:
            conn = psycopg2.connect(
                host = config_postgres['hostname'],
                dbname = config_postgres['database'],
                user = config_postgres['username'],
                password = config_postgres['pwd'],
                port = config_postgres['port_id']
            )
            
            cur = conn.cursor()

            query = """COPY ML_DEV.HOUSING_PROCESSED_TEST_DATA FROM STDIN WITH DELIMITER as '|'"""
            with open('artifacts/test.txt','r') as file:
                next(file)
                cur.copy_expert(
                    query,
                    file
                )
                conn.commit()
                file.close()


        except Exception as e:
              raise CustomException(e, sys)

        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()         


if __name__=="__main__":
   data_etl = DataETL()
   data_etl.process_housing_file()
   data_etl.create_train_table_dev()
   data_etl.create_test_table_dev()
   data_etl.load_train_table_dev()
   data_etl.load_test_table_dev()