from src.exceptions import CustomException
from src.logger import logging

from src.exceptions import CustomException
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from src.components.data_transformation import DataTransformation
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv('notebooks/data/data_no_outliers.csv')
            logging.info("data reading using pandas")


            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok = True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index = False , header = True)
            logging.info("raw data saved")

            train_df,test_df = train_test_split(df,test_size = 0.2,random_state = 42)
            train_df.to_csv(self.data_ingestion_config.train_data_path,index = False,header = True)
            test_df.to_csv(self.data_ingestion_config.test_data_path,index = False,header = True)
            logging.info("train test split and saved")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    print(f"Training data path is : {train_data}")
    print(f"testing data path is :{test_data}")

    data_transformation = DataTransformation()
    train_arr,test_arr,_,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    model_accuracy = model_trainer.initiate_model_trainer(train_arr,test_arr)
    print(model_accuracy)

