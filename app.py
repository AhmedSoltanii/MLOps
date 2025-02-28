import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_pipeline import load_model

app = FastAPI()

# Charger le modèle et ses transformateurs
model, encoder, scaler = load_model()

if model is None:
    raise RuntimeError(
        "Le modèle n'a pas été trouvé. Entraînez et sauvegardez le modèle avant de lancer l'API."
    )


class InputData(BaseModel):
    features: list[float]


@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convertir les données en tableau numpy
        input_data = np.array(data.features).reshape(1, -1)

        # Appliquer la normalisation
        input_data_scaled = scaler.transform(input_data)

        # Effectuer la prédiction
        prediction = model.predict(input_data_scaled)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
