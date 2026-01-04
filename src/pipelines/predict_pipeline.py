import sys
import pandas as pd
from src.exceptions import CustomException
from src.utils import load_object
import os

class CustomData:
    def __init__(self,
        Delay_from_due_date:int ,
        Num_of_Delayed_Payment:int,
        Num_Credit_Inquiries:int,
        Credit_Utilization_Ratio:float,
        Credit_History_Age:int,
        Payment_of_Min_Amount:str,
        Amount_invested_monthly:float,
        Monthly_Balance:float,
        Credit_Mix:str,
        Payment_Behaviour:str,
        Age:int,
        Annual_Income:float,
        Num_Bank_Accounts:int,
        Num_Credit_Card:int,
        Interest_Rate:float,
        Num_of_Loan:int,
        Monthly_Inhand_Salary:float,
        Changed_Credit_Limit:int,
        Outstanding_Debt:float,
        Total_EMI_per_month:float  ):

        self.Delay_from_due_date = Delay_from_due_date
        self.Num_of_Delayed_Payment = Num_of_Delayed_Payment
        self.Num_Credit_Inquiries = Num_Credit_Inquiries
        self.Credit_Utilization_Ratio = Credit_Utilization_Ratio
        self.Credit_History_Age = Credit_History_Age
        self.Payment_of_Min_Amount = Payment_of_Min_Amount
        self.Amount_invested_monthly = Amount_invested_monthly
        self.Monthly_Balance = Monthly_Balance
        self.Credit_Mix = Credit_Mix
        self.Payment_Behaviour = Payment_Behaviour
        self.Age = Age
        self.Annual_Income = Annual_Income
        self.Num_Bank_Accounts = Num_Bank_Accounts
        self.Num_Credit_Card = Num_Credit_Card
        self.Interest_Rate = Interest_Rate
        self.Num_of_Loan = Num_of_Loan
        self.Monthly_Inhand_Salary = Monthly_Inhand_Salary
        self.Changed_Credit_Limit = Changed_Credit_Limit
        self.Outstanding_Debt = Outstanding_Debt
        self.Total_EMI_per_month = Total_EMI_per_month  

    def to_df(self):
        try:

            custom_data_input_dict = {
                "Delay_from_due_date": [self.Delay_from_due_date],
                "Num_of_Delayed_Payment": [self.Num_of_Delayed_Payment],
                "Num_Credit_Inquiries": [self.Num_Credit_Inquiries],
                "Credit_Utilization_Ratio": [self.Credit_Utilization_Ratio],
                "Credit_History_Age": [self.Credit_History_Age],
                "Payment_of_Min_Amount": [self.Payment_of_Min_Amount],
                "Amount_invested_monthly": [self.Amount_invested_monthly],
                "Monthly_Balance": [self.Monthly_Balance],
                "Credit_Mix": [self.Credit_Mix],
                "Payment_Behaviour": [self.Payment_Behaviour],
                "Age": [self.Age],
                "Annual_Income": [self.Annual_Income],
                "Num_Bank_Accounts": [self.Num_Bank_Accounts],
                "Num_Credit_Card": [self.Num_Credit_Card],
                "Interest_Rate": [self.Interest_Rate],
                "Num_of_Loan": [self.Num_of_Loan],
                "Monthly_Inhand_Salary": [self.Monthly_Inhand_Salary],
                "Changed_Credit_Limit": [self.Changed_Credit_Limit],
                "Outstanding_Debt": [self.Outstanding_Debt],
                "Total_EMI_per_month": [self.Total_EMI_per_month]
            }

            dff= pd.DataFrame(custom_data_input_dict)
            return dff

        except Exception as e:
            raise CustomException(e,sys) 

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,data):
        try:

            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","x_transformer.pkl")
            print("Before Loading the model")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            print("After Loading the model")

            print(f"feature Columns : {data.columns}")
            print(f"Data Types : {data.dtypes}")
            print(f"Data Head : {data.head()}")

            data_scaled = preprocessor.transform(data)

            print(f"scaled Data shape : {data_scaled.shape}")
            preds = model.predict(data_scaled)
            
            print(f"prediction : {preds}")
            return preds
        except Exception as e:
            raise CustomException(e,sys)






    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    