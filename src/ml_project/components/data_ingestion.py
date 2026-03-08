import os # for handling file paths
import sys # for handling system-specific parameters and functions
from src.ml_project.exception import CustomException # custom exception handling
from src.ml_project.logger import logging # custom logging
import pandas as pd # for data manipulation and analysis
from src.ml_project.utils import read_sql_data # for reading data from MySQL database
from dataclasses import dataclass # for creating data classes
from sklearn.model_selection import train_test_split # for splitting data into training and testing sets

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv') # path for training data
    test_data_path: str = os.path.join('artifacts', 'test.csv') # path for testing data
    raw_data_path: str = os.path.join('artifacts', 'data.csv') # path for raw data 
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() # initialize the data ingestion configuration
        
    def initiate_data_ingestion(self):
        try:
            df = read_sql_data() # read data from MySQL database
            logging.info("Reading from the mysql database as a dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # create directories if they don't exist
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # save raw data to CSV
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) # split data into training and testing sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # save training data to CSV
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) # save testing data to CSV
            logging.info("Ingestion of the data is completed")
            
            return(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path) # return paths to training and testing data
        except Exception as e:
            raise CustomException(e, sys)