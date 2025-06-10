from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware 
import requests
import re 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "").lower()

    try:
        if "crea" in prompt or "genera" in prompt:
            # Estrai prodotto tra virgolette
            product_match = re.search(r'"(.*?)"', prompt)
            rating_match = re.search(r'([1-5])\s*(?:stelle|stella)', prompt)

            product = product_match.group(1).strip() if product_match else "prodotto sconosciuto"
            rating = int(rating_match.group(1)) if rating_match else 5

            response = requests.post(
                "http://reviews-generator:8001/generate",
                json={"product": product, "rating": rating}
            )
            return response.json()

        elif "valuta" in prompt or "classifica" in prompt:
            # Estrai contenuto tra virgolette doppie
            match = re.search(r'"(.*?)"', prompt)
            if match:
                review_text = match.group(1)
            else:
                review_text = prompt.replace("valuta", "").replace("classifica", "").strip()

            response = requests.post(
                "http://reviews-classifier:8002/classify",
                json={"text": review_text}
            )
            return response.json()

        return {"error": "Prompt non riconosciuto"}

    except Exception as e:
        return {"error": f"Errore durante la richiesta: {str(e)}"}
