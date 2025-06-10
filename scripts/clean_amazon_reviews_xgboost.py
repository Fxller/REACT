import os
import re
import requests
import emoji
import contractions
import pandas as pd
import numpy as np
import fasttext
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.utils import resample

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Carica dataset raw
df = pd.read_csv("data/raw/reviews/amazon_beauty_reviews.csv")

# Uniforma valori nulli
null_equivalents = ["^NaN$", "^na$", "^NA$", "^null$", "^None$", "^N/A$", "^n/a$", "^\s*$"]
df.replace(to_replace=null_equivalents, value=np.nan, regex=True, inplace=True)
df_clean = df[["text", "rating"]].dropna(subset=["text", "rating"])
df_clean = df_clean.reset_index(drop=True)

# Path assoluto calcolato rispetto al file corrente (scripts/clean_amazon_reviews.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
LID_MODEL_PATH = os.path.join(script_dir, "..", "reviews-classifier", "lid.176.ftz")
LID_MODEL_PATH = os.path.abspath(LID_MODEL_PATH)

if not os.path.exists(LID_MODEL_PATH):
    raise FileNotFoundError(f"Il modello FastText '{LID_MODEL_PATH}' non è stato trovato.")

model = fasttext.load_model(LID_MODEL_PATH)

def get_lang(text):
    try:
        prediction = model.predict(str(text))
        lang = prediction[0][0].replace("__label__", "")
        return lang
    except:
        return "unknown"

df_clean = df_clean[df_clean["text"].notna()]
df_clean = df_clean[df_clean["text"].str.len() > 0]

df_clean["lang"] = df_clean["text"].apply(get_lang)

df_clean = df_clean[df_clean["lang"] != "es"]
df_clean = df_clean.drop(columns=["lang"])

# Preprocessing testo
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Rimozione emoji
    text = emoji.replace_emoji(text, replace='')
    
    # 3. Rimozione URL e HTML
    text = re.sub(r"http\S+|www\S+|<.*?>", " ", text)
    
    # 4. Rimozione punteggiatura (ma lascia apostrofi utili)
    text = re.sub(r"[^\w\s']", " ", text)
    
    # 5. Rimozione numeri
    text = re.sub(r"\d+", " ", text)
    
    # 6. Rimozione stopword
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # 7. Ricomponi e rimuovi spazi multipli
    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

df_clean["text_cleaned"] = df_clean["text"].astype(str).apply(preprocess_text)
# Espansione contrazioni
def expand_contractions(text):
    return contractions.fix(text)

df_clean["text_expanded"] = df_clean["text_cleaned"].apply(contractions.fix)

# Marcatura negazioni
def mark_negations(text):
    negation_words = {"not", "no", "never", "n't"}
    words = text.split()
    result, negating = [], False
    for word in words:
        if any(n in word for n in negation_words):
            negating = True
            result.append(word)
        elif negating:
            result.append("NOT_" + word)
            if any(p in word for p in [".", ",", "!", "?", ";"]):
                negating = False
        else:
            result.append(word)
    return " ".join(result)

df_clean["text_final"] = df_clean["text_expanded"].apply(mark_negations)

df_clean = df_clean.drop_duplicates(subset=["text_final"])

# Feature engineering
df_clean["polarity_final"] = df_clean["text_final"].apply(lambda x: TextBlob(x).sentiment.polarity)
df_clean["text_len"] = df_clean["text_final"].apply(lambda x: len(x.split()))

q3 = df_clean[df_clean["rating"] == 3]["polarity_final"].quantile([0.25, 0.75])
iqr3 = q3[0.75] - q3[0.25]
lower3 = q3[0.25] - 1.5 * iqr3
upper3 = q3[0.75] + 1.5 * iqr3

def is_incoherent(row):
    r, p = row["rating"], row["polarity_final"]
    if r in [1, 2] and p >= 0:
        return True
    elif r == 3 and not (lower3 <= p <= upper3):
        return True
    elif r in [4, 5] and p <= 0:
        return True
    return False

df_clean["incoerente"] = df_clean.apply(is_incoherent, axis=1)
df_clean = df_clean[(df_clean["incoerente"] == False) & (df_clean["text_len"] >= 3)].copy()
df_clean = df_clean.drop(columns=["incoerente"])

# Classi target
def map_rating_to_sentiment(rating):
    if rating in [1, 2]:
        return 0
    elif rating == 3:
        return 1
    else:  # rating 4, 5
        return 2
    
df_clean["sentiment_class"] = df_clean["rating"].apply(map_rating_to_sentiment)

# Bilanciamento
target_count = df_clean[df_clean["sentiment_class"] == 0].shape[0]
n_clusters = 5
balanced_parts = [df_clean[df_clean["sentiment_class"] == 0].copy()]

for cls in [1, 2]:
    subset = df_clean[df_clean["sentiment_class"] == cls].copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    subset["cluster"] = kmeans.fit_predict(subset[["polarity_final", "text_len"]])
    samples_per_cluster = target_count // n_clusters
    sampled_clusters = []

    for c in range(n_clusters):
        cluster_df = subset[subset["cluster"] == c]
        sample = resample(cluster_df, 
                          replace=False, 
                          n_samples=min(samples_per_cluster, len(cluster_df)),
                          random_state=42)
        sampled_clusters.append(sample)

    total_current = sum(len(df) for df in sampled_clusters)
    if total_current < target_count:
        remaining = subset[~subset.index.isin(pd.concat(sampled_clusters).index)]
        additional = resample(remaining, 
                         replace=False,
                         n_samples=(target_count - total_current),
                         random_state=42)
        sampled_clusters.append(additional)
    balanced_parts.append(pd.concat(sampled_clusters))

balanced_df = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

balanced_df = balanced_df[["text_final", "sentiment_class", "polarity_final", "text_len"]].copy()

# Salva il dataset pulito
os.makedirs("data/clean", exist_ok=True)
balanced_df.to_csv("data/clean/amazon_beauty_reviews_clean.csv", index=False)
print("Dati puliti salvati in data/clean/amazon_beauty_reviews_clean.csv")
