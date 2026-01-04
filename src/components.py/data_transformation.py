from src.exceptions import CustomException
from src.logger import logging

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass
from src.utils import save_object

@dataclass 
class DataTransformationConfig:
    X_data_transformation_path:str =os.path.join('artifacts','x_transformer.pkl')
    Y_data_transformation_path:str =os.path.join('artifacts','y_transformer.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformer object started fot training data")
            numerical_columns = ['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Credit_Utilization_Ratio', 
            'Credit_History_Age', 'Amount_invested_monthly', 
            'Monthly_Balance', 'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Monthly_Inhand_Salary', 
            'Changed_Credit_Limit', 'Outstanding_Debt', 'Total_EMI_per_month']

            X_cat_columns = ['Payment_of_Min_Amount', 'Credit_Mix', 'Payment_Behaviour']
            target_column = ['Credit_Score']


            num_pipeline = Pipeline(

                steps = [
                    
                    ('imputer',SimpleImputer(strategy = 'median')),
                    ('scaler',StandardScaler())
                ]
            )

            X_cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy = 'most_frequent')),
                    ('onehot',OneHotEncoder(handle_unknown='ignore')),
                    ("standard_scaler" , StandardScaler(with_mean = False)) 
                ]
            )

            
            X_preprocessor = ColumnTransformer(
                transformers=[
                            ("numerical_pipeline",num_pipeline,numerical_columns),
                            ("X_categorical_pipeline",X_cat_pipeline,X_cat_columns)
                ],remainder='drop'
                            )
            Y_preprocessor = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal', OrdinalEncoder(categories=[['Poor','Standard','Good']]))
                ]   
            )

                
                            
            return X_preprocessor,Y_preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("data Transformation initiated")

            train_df = pd.read_csv(train_path)
            test_df =  pd.read_csv(test_path)


            
            logging.info("read Train and test data comleted")

            logging.info("obtaining a preprocessing obj")

            X_preprocessing_obj,Y_preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'Credit_Score'

            input_feature_train_df = train_df.drop(columns = [target_column_name])
            target_feature_train_df  = train_df[[target_column_name]]

            input_feature_test_df = test_df.drop(columns = [target_column_name])
            target_feature_test_df  = test_df[[target_column_name]]


            logging.info("Appling Preprocessing obh=ject ")


            input_feature_train_arr  = X_preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = X_preprocessing_obj.transform(input_feature_test_df)


            target_feature_train_arr = Y_preprocessing_obj.fit_transform(target_feature_train_df)
            target_feature_test_arr = Y_preprocessing_obj.transform(target_feature_test_df)

            y_feature_train_arr = target_feature_train_arr.ravel()
            y_feature_test_arr = target_feature_test_arr.ravel()
            
            train_arr=  np.c_[input_feature_train_arr,y_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr,y_feature_test_arr]

            os.makedirs(os.path.dirname(self.data_transformation_config.X_data_transformation_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.Y_data_transformation_path), exist_ok=True)


            save_object(file_path=self.data_transformation_config.X_data_transformation_path , obj = X_preprocessing_obj)
            save_object(file_path=self.data_transformation_config.Y_data_transformation_path , obj = Y_preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.X_data_transformation_path,
                self.data_transformation_config.Y_data_transformation_path
            )
        except Exception as e:
            raise CustomException(e,sys)

            

 


