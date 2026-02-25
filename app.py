import sys
import os


from src.ml_project.logger import logging
from src.ml_project.exception import CustomException



if __name__ == "__main__":
    logging.info("The execution of the main function has started.")
    
    
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("An error occurred in the main function.")
        raise CustomException(e, sys)