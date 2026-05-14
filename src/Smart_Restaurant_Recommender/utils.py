import numpy as np
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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


def build_user_embedding(selected_categories,selected_extras,selected_atmospheres):
    tokens = []

    if selected_categories:
        tokens.extend(selected_categories)
    
    if selected_extras:
        tokens.extend(selected_extras)

    if selected_atmospheres:
        tokens.extend(selected_atmospheres)
        tokens.append("atmosphere")

    text = ", ".join(tokens)

    print(text)

    return encode_text(text)




def get_initial_recos(user_emb, X_text, df, top_n=10):

    user_emb = np.asarray(user_emb)

    # FIX: ensure 2D shape (1, dim)
    if user_emb.ndim == 1:
        user_emb = user_emb.reshape(1, -1)

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


def encode_text(text):
    return MODEL.encode(text, normalize_embeddings=True)

def ambience_to_tokens(amb):
    if not isinstance(amb, dict):
        return []
    return [k for k, v in amb.items() if v is True]


def build_full_text(row):
    categories = str(row.get("description", "")).lower()
    category_tokens = [w.strip() for w in categories.split(",") if w.strip()]
    boosted_categories = ", ".join(category_tokens)

    amb_tokens = ambience_to_tokens(row.get("Ambience_parsed", None))
    amb_str = (" " + " ".join(amb_tokens) + " atmosphere") if amb_tokens else ""

    return boosted_categories + amb_str


def clean_text(t):
    seen = set()
    tokens = []
    for w in t.split(","):
        w = w.strip().lower()
        if w and w not in seen:
            seen.add(w)
            tokens.append(w)
    return " ".join(tokens)

def parse_ambience(val):
    if val is None:
        return None
    attr = val if isinstance(val, dict) else ast.literal_eval(val)
    ambience = attr.get('Ambience')
    if ambience is None:
        return None
    if isinstance(ambience, dict):
        return ambience
    # Si c'est une string, on parse
    return ast.literal_eval(ambience)
    