import sys
import os
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import (

    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from src.utils import evaluate_models
from src.exceptions import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score
from src.utils import save_object , load_object

@dataclass
class ModelTrainerConfig:
    trainer_model_path:str = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("initiate model Trainer")


            X_train,y_train,X_test,y_test = (

            train_arr[:,:-1],
            train_arr[:,-1],
            test_arr[:,:-1],
            test_arr[:,-1]
            )

            models = {

               
                "LogisticRegression":LogisticRegression(),
                "AdaBoostClassifier":AdaBoostClassifier(),
                "GradientBoostingClassifier":GradientBoostingClassifier(),
                "RandomForestClassifier":RandomForestClassifier(),
                "KNeigboursClassifier":KNeighborsClassifier(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "XGBClassifier":GradientBoostingClassifier()

            }
            model_report:dict = evaluate_models(X_train= X_train , y_train = y_train ,
                                         X_test = X_test , y_test = y_test,models = models)

            
            ## To get best model Score from report 3
            best_model_score=  max(sorted(model_report.values()))
           
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model found with accuracy at least 60%")

            logging.info(f"Best model found with accuracy: {best_model_name}  : {best_model_score}")

            
            save_object(
                file_path  = self.model_trainer_config.trainer_model_path,
                obj  = best_model

            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test,predicted)

            logging.info("Best model found with accuracy: {best_model_name}  : {accuracy}")
            return accuracy

        except Exception as e:
            raise CustomException(e,sys)



























