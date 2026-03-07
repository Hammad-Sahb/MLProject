import os # for handling file paths
import sys # for handling system-specific parameters and functions
from src.ml_project.exception import CustomException # custom exception handling
from src.ml_project.logger import logging # custom logging
import pandas as pd # for data manipulation and analysis
from dotenv import load_dotenv # for loading environment variables
import pymysql # for connecting to MySQL database

load_dotenv() # Load environment variables from .env file

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("database")

def read_sql_data():
    try:
        logging.info("Reading data from the mysql database as a dataframe")
        # Here you would add the code to connect to your MySQL database and read the data into a DataFrame
        mydb = pymysql.connect(host=host, user=user, password=password, database=database)
        logging.info("Connecting to the MySQL database", mydb)
        df=pd.read_sql("SELECT * from students", mydb)
        print(df.head())
        
        return df
        # Forexample, you could use SQLAlchemy or pymysql to connect to the database and execute a query
        # Then you would return the DataFrame containing the data
    except Exception as e:
        raise CustomException(e, sys)