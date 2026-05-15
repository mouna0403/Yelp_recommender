import ast

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def build_user_embedding(selected_categories, selected_extras, selected_atmospheres):
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


def get_initial_recos(
    user_emb, X_text, df, top_n=10, selected_price=None, selected_city=None
):

    user_emb = np.asarray(user_emb)

    if user_emb.ndim == 1:
        user_emb = user_emb.reshape(1, -1)

    sims = cosine_similarity(user_emb, X_text).flatten()

    df_filtered = df.copy()

    # ── PRICE FILTER ─────────────────────
    if selected_price is not None:
        price = pd.to_numeric(df_filtered["RestaurantsPriceRange2"], errors="coerce")
        df_filtered = df_filtered[price == selected_price]

    # ── CITY FILTER ───────────────────────
    if selected_city is not None:
        df_filtered = df_filtered[
            df_filtered["city"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq(str(selected_city).lower())
        ]

    # ── ALIGN SIMS AFTER FILTER ───────────
    sims = sims[df_filtered.index]

    top_idx = np.argsort(sims)[::-1][:top_n]

    return df_filtered.iloc[top_idx]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
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
    ambience = attr.get("Ambience")
    if ambience is None:
        return None
    if isinstance(ambience, dict):
        return ambience
    # Si c'est une string, on parse
    return ast.literal_eval(ambience)


def extract_price_range(attr):
    if attr is None:
        return None

    # si string → dict
    if isinstance(attr, str):
        try:
            attr = ast.literal_eval(attr)
        except:
            return None

    if isinstance(attr, dict):
        return attr.get("RestaurantsPriceRange2")

    return None


def parse_value(v):
    if not isinstance(v, str):
        return v
    pattern = r'\bu([\'"])'
    cleaned = re.sub(pattern, r"\1", v.strip())
    try:
        return ast.literal_eval(cleaned)
    except:
        return cleaned


def flatten_attributes(attr_dict):
    tokens = []

    def process(key, raw_val):
        val = parse_value(raw_val)

        # sous-dict : appliquer les mêmes règles sur chaque sous-clé
        if isinstance(val, dict):
            for sub_k, sub_v in val.items():
                process(sub_k, sub_v)
            return

        # False / None → supprimer
        if val in {False, None}:
            return
        if isinstance(val, str) and val.lower() in ("false", "none", "no"):
            return

        # True → juste la clé
        if val is True:
            tokens.append(key)
            return
        if isinstance(val, str) and val.lower() in ("true", "yes"):
            tokens.append(key)
            return

        # numérique ou string → "key val"
        if isinstance(val, (int, float)):
            tokens.append(f"{key}: {val}")
            return

        clean = val.strip("'\"")
        if clean.lower() not in ("false", "none", "no", ""):
            tokens.append(f"{key} {clean}")

    for k, v in attr_dict.items():
        process(k, v)

    return tokens


import re


def clean_token(token):
    token = token.replace("_", " ")
    # Exception WiFi avant de splitter les majuscules
    token = token.replace("WiFi", "Wifi")
    token = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
    token = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", token)
    token = re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", token)
    token = token.replace("Wifi", "WiFi")
    return token


def build_attribute_text(x):
    if not isinstance(x, dict):
        return ""
    tokens = flatten_attributes(x)
    return "\n".join(clean_token(t) for t in tokens)
