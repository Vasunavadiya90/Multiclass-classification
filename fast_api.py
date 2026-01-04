import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
from src.utils import load_object
from src.pipelines.predict_pipeline import PredictPipeline,CustomData
from src.logger import logging

app = FastAPI(title="Credit Card Default Prediction API",
              description="API for Credit Card Default Prediction")

class InputData(BaseModel):
    Delay_from_due_date:int 
    Num_of_Delayed_Payment:int
    Num_Credit_Inquiries:int
    Credit_Utilization_Ratio:float
    Credit_History_Age:int
    Payment_of_Min_Amount:str
    Amount_invested_monthly:float
    Monthly_Balance:float
    Credit_Mix:str
    Payment_Behaviour:str
    Age:int
    Annual_Income:float
    Num_Bank_Accounts:int
    Num_Credit_Card:int
    Interest_Rate:float
    Num_of_Loan:int
    Monthly_Inhand_Salary:float
    Changed_Credit_Limit:float
    Outstanding_Debt:float
    Total_EMI_per_month:float


@app.post('/predict')
async def predict(data:InputData):
    try:
        logging.info("Starting the prediction pipeline")
        custom_data = CustomData(
            Delay_from_due_date=  data.Delay_from_due_date,
            Num_of_Delayed_Payment=data.Num_of_Delayed_Payment,
            Num_Credit_Inquiries=data.Num_Credit_Inquiries,
            Credit_Utilization_Ratio=data.Credit_Utilization_Ratio,
            Credit_History_Age=data.Credit_History_Age,
            Payment_of_Min_Amount=data.Payment_of_Min_Amount,
            Amount_invested_monthly=data.Amount_invested_monthly,
            Monthly_Balance=data.Monthly_Balance,
            Credit_Mix=data.Credit_Mix,
            Payment_Behaviour=data.Payment_Behaviour,
            Age=data.Age,
            Annual_Income=data.Annual_Income,
            Num_Bank_Accounts=data.Num_Bank_Accounts,
            Num_Credit_Card=data.Num_Credit_Card,
            Interest_Rate=data.Interest_Rate,
            Num_of_Loan=data.Num_of_Loan,
            Monthly_Inhand_Salary=data.Monthly_Inhand_Salary,
            Changed_Credit_Limit=data.Changed_Credit_Limit,
            Outstanding_Debt=data.Outstanding_Debt,
            Total_EMI_per_month=data.Total_EMI_per_month
        )

        pred_df = custom_data.to_df()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        logging.info(f"Prediction completed {results}")

        return {
            "prediction:":float(results[0])
        }
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app , host='127.0.0.1',port = 8000)
    