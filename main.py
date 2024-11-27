import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import pandas as pd
from helper import convert_to_month_name, transform_new_data

# Load the model and encoder
model = joblib.load('model.pkl')  
encoder = joblib.load('encoder.pkl')
original_one_hot_columns = joblib.load('original_one_hot_columns.pkl')

app = FastAPI()

# Pydantic model for input data validation
class Item(BaseModel):
    MONATSZAHL: str
    AUSPRAEGUNG: str
    JAHR: str
    MONAT: str
# endpoint for inference
@app.post("/predict/")
async def predict(item: Item):
    try:

        input_data = {
            "MONATSZAHL": item.MONATSZAHL,
            "AUSPRAEGUNG": item.AUSPRAEGUNG,
            "JAHR": item.JAHR,
            "MONAT": item.MONAT
        }


        for key, value in input_data.items():
            if not value or value.strip() == "":
                raise HTTPException(status_code=400, detail=f"{key} cannot be empty")


        input_df = pd.DataFrame([input_data])


        input_df['MONAT'] = input_df['MONAT'].apply(convert_to_month_name)


        transform_new_data(input_df, encoder, ['MONAT', 'AUSPRAEGUNG', 'JAHR', 'MONATSZAHL'])


        prediction = model.predict(input_df)

        return {"prediction": prediction.tolist()}

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected column: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

