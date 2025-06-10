from datasets import load_dataset
import pandas as pd
import os

# Scarica il dataset recensioni
dataset_reviews = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023", 
    "raw_review_All_Beauty", 
    split="full"
)

# Scarica il dataset metadati
dataset_metadati = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023", 
    "raw_meta_All_Beauty", 
    split="full"
)

# Converti in DataFrame
df_reviews = dataset_reviews.to_pandas()
df_metadati = dataset_metadati.to_pandas()

# Crea directory se non esistono
os.makedirs("data/raw/reviews", exist_ok=True)
os.makedirs("data/raw/metadati", exist_ok=True)

# Salva i CSV
df_reviews.to_csv("data/raw/reviews/amazon_beauty_reviews.csv", index=False)
df_metadati.to_csv("data/raw/metadati/amazon_beauty_metadati.csv", index=False)

print("Dataset recensioni salvato in data/raw/reviews/amazon_beauty_reviews.csv")
print("Dataset metadati salvato in data/raw/metadati/amazon_beauty_metadati.csv")
