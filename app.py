import sys
import os


from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.components.data_ingestion import DataIngestionConfig



if __name__ == "__main__":
    logging.info("The execution of the main function has started.")
    
    
    try:
        #DataIngestionConfig = DataIngestionConfig()
        DataIngestion = DataIngestion()
        DataIngestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("An error occurred in the main function.")
        raise CustomException(e, sys)