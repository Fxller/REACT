from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import uvicorn
from textblob import TextBlob
import numpy as np
from scipy.sparse import hstack, csr_matrix 
import logging 

# === Configura logging ===
logging.basicConfig(
    filename="inference.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Percorsi dei file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost", "xgb_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "xgboost", "vectorizer.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "xgboost", "scaler.pkl")

# Carica oggetti
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)

# Inizializza app
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/classify")
async def classify(input_data: InputText):
    text = input_data.text

    # TFIDF
    X_text = vectorizer.transform([text])

    # Ricostruisci le due feature
    polarity = TextBlob(text).sentiment.polarity
    length = len(text.split())

    # Normalizza
    extra = scaler.transform([[polarity, length]])
    extra_sparse = csr_matrix(extra)

    # Combina
    X_combined = hstack([X_text, extra_sparse])

    # Predizione
    rating = model.predict(X_combined)[0]
    
    logging.info(f"Classify | Input: {text} => Prediction: {rating}")
    
    return {"rating": int(rating)}