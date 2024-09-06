# Crop Prediction API

This project is a FastAPI-based machine learning model that predicts the most suitable crop based on temperature, humidity, and rainfall input. The API is deployed and can be consumed by mobile or web applications for crop prediction purposes.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [License](#license)

## Features

- Predicts suitable crops based on environmental parameters.
- RESTful API built with FastAPI.
- Machine learning model using `scikit-learn` for classification.
- Input: Temperature, Humidity, Rainfall.
- Outputs: Predicted crop label.
- Includes scaling of input data and label encoding.

## Requirements

- Python 3.8+
- FastAPI
- `scikit-learn`
- `numpy`
- `uvicorn`
- `pydantic`
- `pickle`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/crop-prediction-api.git
    cd crop-prediction-api
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Place your pre-trained `model.pkl`, `scaler.pkl`, and `label_encoder.pkl` files in the root directory of the project.

## Usage

1. Run the FastAPI app locally using `uvicorn`:

    ```bash
    uvicorn main:app --reload
    ```

2. The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

### POST `/predict`

- **Description**: Predicts the most suitable crop based on the provided temperature, humidity, and rainfall.
- **Request Body**:
  ```json
  {
    "temperature": 25.0,
    "humidity": 80.0,
    "rainfall": 200.0
  }

