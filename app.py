import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_pipeline import load_model

app = FastAPI()

# Load model, encoder, and scaler
model, encoder, scaler = load_model()

if model is None:
    raise RuntimeError("Le modèle n'a pas été trouvé. Entraînez et sauvegardez le modèle avant de lancer l'API.")

# Define input data schema
class InputData(BaseModel):
    State: str
    Account_length: int
    Area_code: int
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: int
    Total_day_minutes: float
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_minutes: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_minutes: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_minutes: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert input data to DataFrame for encoding
        categorical_data = pd.DataFrame({
            "State": [data.State], 
            "International plan": [data.International_plan], 
            "Voice mail plan": [data.Voice_mail_plan]
        })

        # Encode categorical features (Result is already 2D)
        encoded_data = encoder.transform(categorical_data).to_numpy()

        # Prepare numerical features (Convert to 2D array)
        numeric_data = np.array([[
            data.Account_length, data.Area_code, data.Number_vmail_messages,
            data.Total_day_minutes, data.Total_day_calls, data.Total_day_charge,
            data.Total_eve_minutes, data.Total_eve_calls, data.Total_eve_charge,
            data.Total_night_minutes, data.Total_night_calls, data.Total_night_charge,
            data.Total_intl_minutes, data.Total_intl_calls, data.Total_intl_charge,
            data.Customer_service_calls
        ]])

        # Ensure correct concatenation (Both arrays must be 2D)
        full_input = np.concatenate((encoded_data, numeric_data), axis=1)

        # Normalize input
        input_data_scaled = scaler.transform(full_input)

        probability = model.predict_proba(input_data_scaled)[0][1]

        return {
            "churn_probability": f"{round(probability * 100, 2)}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
