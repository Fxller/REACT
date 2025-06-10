from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import logging

# === Configura logging ===
logging.basicConfig(
    filename="inference.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Caricamento modello
MODEL_DIR = os.path.join("t5-small-finetuned", "final_model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# === FastAPI app
app = FastAPI()

class InputData(BaseModel):
    product: str
    rating: int

@app.post("/generate")
async def generate(data: InputData):
    product = data.product
    rating = data.rating

    # Costruisci il prompt per il modello
    prompt = f"Generate a review of '{product}' with a rating of {rating} stars."
    print(f"Prompt: {prompt}")

    # Tokenizza e genera
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.5,  
        repetition_penalty=1.8   
    )
    review = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    logging.info(f"Generate | Product: {product}, Rating: {rating} => Review: {review}")

    return {"review": review}

