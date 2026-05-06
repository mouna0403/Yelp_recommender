import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer




MODEL = SentenceTransformer("all-MiniLM-L6-v2")




def extract_categories(df):
    cats = (
        df["description"]
        .fillna("")
        .apply(lambda x: [c.strip() for c in x.split(",") if c.strip()])
        .explode()
        .dropna()
        .unique()
    )
    return sorted(cats)


def build_user_embedding(categories):
    text = clean_text(", ".join(categories))

    return MODEL.encode(
        [text],
        normalize_embeddings=True
    )


def get_initial_recos(user_emb, X_text, df, top_n=10):
    sims = cosine_similarity(user_emb, X_text).flatten()
    top_idx = np.argsort(sims)[::-1][:top_n]

    return df.iloc[top_idx]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def clean_text(t):
    return " ".join(
        sorted(set(
            w.strip().lower()
            for w in t.split(",")
            if w.strip()
        ))
    )

def encode_text(text):
    return MODEL.encode(text, normalize_embeddings=True)