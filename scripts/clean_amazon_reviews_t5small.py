from datasets import load_dataset
import pandas as pd
import re
import numpy as np
from datasets import Dataset
import matplotlib.pyplot as plt

# Caricamento del dataset, considerando solo raw review All Beauty

df = pd.read_csv("data/raw/reviews/amazon_beauty_reviews.csv")

# 1. Rimpiazza tutte le forme testuali di null con np.nan
for col in ["text", "title"]:
    df[col] = df[col].replace(to_replace=["nan", "NaN", "None", "null", "NULL", ""], value=np.nan)

# 2. Rimuovi righe con valori nulli reali
df = df.dropna(subset=["text", "title", "rating"])

# 3. Convertili in stringa (in caso qualcosa sia ancora float o altro)
df["text"] = df["text"].astype(str)
df["title"] = df["title"].astype(str)

# 4. Elimina righe con solo spazi (o vuote anche dopo strip)
df = df[(df["text"].str.strip() != "") & (df["title"].str.strip() != "")]

# 5. (Opzionale) Rimuovi righe con lunghezza < 5 caratteri
df = df[(df["text"].str.len() > 5) & (df["title"].str.len() > 3)]

# 6. Reset dell'indice
df.reset_index(drop=True, inplace=True)

# Rimozione duplicati sul campo 'text' (case insensitive)
df["text_clean"] = df["text"].str.strip().str.lower()
df = df.drop_duplicates(subset=["text_clean"])
df = df.drop(columns=["text_clean"])

df_metadati = pd.read_csv("data/raw/metadati/amazon_beauty_metadati.csv")

df_merged = df.merge(df_metadati, on="parent_asin", how="left")

df_merged = df_merged.rename(columns={"title_y": "product_name"})
df_merged = df_merged.rename(columns={"title_x": "title"})

# Prendo solo le colonne di interesse
df_merged = df_merged[["rating", "product_name", "title", "text"]]

## Far diventare tutto minuscolo
df_merged.loc[:, 'title'] = df_merged['title'].str.lower()
df_merged.loc[:, 'text'] = df_merged['text'].str.lower()
df_merged.loc[:, 'product_name'] = df_merged['product_name'].str.lower()

import numpy as np
import pandas as pd

def stratified_undersample(df, group_col, text_col, target_count):
    result = []

    for label, group in df.groupby(group_col):
        group = group.copy()
        group["length"] = group[text_col].apply(len)
        group_sorted = group.sort_values("length")

        # Definisci indici
        short_range = group_sorted.iloc[:len(group_sorted)//3]
        medium_range = group_sorted.iloc[len(group_sorted)//3:2*len(group_sorted)//3]
        long_range = group_sorted.iloc[2*len(group_sorted)//3:]

        # Calcola le dimensioni con fallback al min(len, target)
        n_short = min(int(target_count * 0.3), len(short_range))
        n_medium = min(int(target_count * 0.4), len(medium_range))
        n_long = min(target_count - n_short - n_medium, len(long_range))

        short = short_range.sample(n=n_short, random_state=42)
        medium = medium_range.sample(n=n_medium, random_state=42)
        long = long_range.sample(n=n_long, random_state=42)

        balanced = pd.concat([short, medium, long])
        result.append(balanced)

    return pd.concat(result).drop(columns=["length"]).reset_index(drop=True)

# Usa il minimo tra le classi
min_count = df_merged["rating"].value_counts().min()
df_balanced = stratified_undersample(df_merged, group_col="rating", text_col="text", target_count=min_count)


df_balanced.loc[:, "input_text"] = df_balanced.apply(
    lambda row: f"Product: {row['product_name']}. Rating: {int(row['rating'])} stars.", axis=1)

df_balanced.loc[:, "target_text"] = df_balanced.apply(
    lambda row: f"Title: {row['title']}\nReview: {row['text']}", axis=1)

# Tieni solo queste due colonne
dataset = df_balanced[["input_text", "target_text"]]

dataset.to_csv("data/clean/amazon_beauty_reviews_clean_t5.csv")