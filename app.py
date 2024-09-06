from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()


model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')


class PredictionRequest(BaseModel):
    temperature: float
    humidity: float
    rainfall: float
    label: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the NPK Prediction API"}


@app.post("/predict/")
def predict_npk(data: PredictionRequest):
    
    temperature = data.temperature
    humidity = data.humidity
    rainfall = data.rainfall
    label = data.label

    
    label_encoded = label_encoder.transform([label])[0]
    numerical_input_data = np.array([[temperature, humidity, rainfall]])

    scaled_numerical_data = scaler.transform(numerical_input_data)

    input_data = np.concatenate([scaled_numerical_data, np.array([[label_encoded]])], axis=1)

    prediction = model.predict(input_data)

   
    return {
        "N": prediction[0][0],
        "P": prediction[0][1],
        "K": prediction[0][2]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

