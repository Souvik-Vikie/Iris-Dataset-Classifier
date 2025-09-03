from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing; later restrict to your Streamlit domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once (not on every request)
MODEL_DIR = Path(__file__).parent / "models"
model_dt = joblib.load(MODEL_DIR / "DecisionTree.pkl")
model_knn = joblib.load(MODEL_DIR / "KNN.pkl")
model_lr = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")

# Define request schema
class Features(BaseModel):
    data: List[float]  # expects {"data": [5.1, 3.5, 1.4, 0.2]}

@app.post("/predict/DecisionTree")
async def predict_dt(payload: Features):
    prediction = model_dt.predict([payload.data])[0]
    return {"model": "DecisionTree", "prediction": int(prediction)}

@app.post("/predict/KNN")
async def predict_knn(payload: Features):
    prediction = model_knn.predict([payload.data])[0]
    return {"model": "KNN", "prediction": int(prediction)}

@app.post("/predict/LogisticRegression")
async def predict_lr(payload: Features):
    prediction = model_lr.predict([payload.data])[0]
    return {"model": "LogisticRegression", "prediction": int(prediction)}



# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
# import joblib
# from pathlib import Path

# app = FastAPI()

# # Load models once (not on every request)
# MODEL_DIR = Path(__file__).parent / "models"
# model_dt = joblib.load(MODEL_DIR / "DecisionTree.pkl")
# model_knn = joblib.load(MODEL_DIR / "KNN.pkl")
# model_lr = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")

# # Define request schema
# class Features(BaseModel):
#     data: List[float]  # expects {"data": [5.1, 3.5, 1.4, 0.2]}

# @app.post("/predict/DecisionTree")
# async def predict_dt(payload: Features):
#     prediction = model_dt.predict([payload.data])[0]
#     return {"model": "DecisionTree", "prediction": int(prediction)}

# @app.post("/predict/KNN")
# async def predict_knn(payload: Features):
#     prediction = model_knn.predict([payload.data])[0]
#     return {"model": "KNN", "prediction": int(prediction)}

# @app.post("/predict/LogisticRegression")
# async def predict_lr(payload: Features):
#     prediction = model_lr.predict([payload.data])[0]
#     return {"model": "LogisticRegression", "prediction": int(prediction)}
