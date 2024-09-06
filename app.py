from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

class CropPredictionRequest(BaseModel):
    temperature: float
    humidity: float
    rainfall: float

@app.post('/predict')
def predict_crop(request: CropPredictionRequest):
    
    data = np.array([[request.temperature, request.humidity, request.rainfall]])
    data = scaler.transform(data)
    

    prediction = model.predict(data)[0]
    
    
    crop = le.inverse_transform([prediction])[0]
    
    return {'predicted_crop': crop}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

