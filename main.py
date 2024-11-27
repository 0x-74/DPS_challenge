import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import pandas as pd
from helper import convert_to_month_name, transform_new_data

# Load the model and encoder
model = joblib.load('model.pkl')  
encoder = joblib.load('encoder.pkl')

app = FastAPI()

# Pydantic model for input data validation
class Item(BaseModel):
    MONATSZAHL: str
    AUSPRAEGUNG: str
    JAHR: int
    MONAT: str

# Endpoint for inference
@app.post("/predict/")
async def predict(item: Item):
    try:
        # Construct input data from request
        input_data = {
            "MONATSZAHL": item.MONATSZAHL,
            "AUSPRAEGUNG": item.AUSPRAEGUNG,
            "JAHR": item.JAHR,
            "MONAT": item.MONAT
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert 'MONAT' to month name
        try:
            input_df['MONAT'] = input_df['MONAT'].apply(convert_to_month_name)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error converting 'MONAT' to month name: {e}"
            )

        # Transform data with encoder
        try:
            transformed_df = transform_new_data(
                input_df, 
                encoder, 
                original_one_hot_columns=['MONATSZAHL', 'AUSPRAEGUNG', "JAHR", 'MONAT']
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error transforming data: {e}"
            )

        # Ensure the transformed data matches the model's expected input
        try:
            prediction = model.predict(transformed_df)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error during model prediction: {e}"
            )

        # Return prediction result
        return {"prediction": prediction.tolist()}

    except ValidationError as e:
        raise HTTPException(
            status_code=422, 
            detail=f"Validation error: {e}"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing expected column: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {e}"
        )
