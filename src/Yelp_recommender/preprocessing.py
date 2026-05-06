import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from Yelp_recommender.utils import MODEL, clean_text
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# ----------------------------
# CACHE GLOBAL (IMPORTANT)
# ----------------------------
_cache = {}

def load_preprocessed():
    if "done" in _cache:
        return _cache["data"]

    print("=== START PREPROCESSING ===")

    df_final = pd.read_parquet(DATA_DIR / "Yelp_dataset_reviews_enriched.parquet")

    mlb = MultiLabelBinarizer()

    cat_df = pd.DataFrame(
        mlb.fit_transform(df_final["macro_categories"].fillna("[]")),
        columns=mlb.classes_,
        index=df_final.index
    )

    cat_df = pd.concat([df_final.drop(columns=["macro_categories"]), cat_df], axis=1)

    df_food = cat_df[cat_df["Food & Beverage"] == 1]
    df_food_business = df_food.drop_duplicates(subset="business_id").reset_index(drop=True)

    tqdm.pandas()
    texts = df_food_business["description"].fillna("").progress_apply(clean_text).tolist()

    X_text = MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    TEXT_SIM = cosine_similarity(X_text)

    INDICES = pd.Series(df_food_business.index, index=df_food_business["business_id"])

    data = df_food_business, X_text, TEXT_SIM, INDICES

    _cache["done"] = True
    _cache["data"] = data

    return data