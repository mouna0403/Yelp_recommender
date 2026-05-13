from Yelp_recommender.utils import MODEL, clean_text

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import joblib

print("=== START PREPROCESSING ===")

# ----------------------------
# LOAD DATA
# ----------------------------
print("[1] Loading data...")
df_final = pd.read_parquet("data/Yelp_dataset_reviews_enriched.parquet")
print(f"Loaded: {df_final.shape}")

# ----------------------------
# MULTI LABEL ENCODING
# ----------------------------
print("[2] Encoding categories...")

mlb = MultiLabelBinarizer()

cat_df = pd.DataFrame(
    mlb.fit_transform(df_final['macro_categories'].fillna('[]')),
    columns=mlb.classes_,
    index=df_final.index
)

cat_df = pd.concat(
    [df_final.drop(columns=['macro_categories']), cat_df],
    axis=1
)

print(f"Encoded shape: {cat_df.shape}")

# ----------------------------
# FILTER FOOD
# ----------------------------
print("[3] Filtering Food & Beverage...")

df_food = cat_df[cat_df['Food & Beverage'] == 1]
df_food_business = df_food.drop_duplicates(subset='business_id').reset_index(drop=True)

print(f"Food businesses: {df_food_business.shape}")

# ----------------------------
# TEXT CLEANING
# ----------------------------
print("[4] Cleaning text...")

tqdm.pandas()

texts = df_food_business["description"].fillna("").progress_apply(clean_text).tolist()

print(f"Texts ready: {len(texts)}")

# ----------------------------
# EMBEDDINGS
# ----------------------------
print("[5] Encoding embeddings...")

model = MODEL

X_text = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

print(f"Embeddings shape: {X_text.shape}")

# ----------------------------
# SIMILARITY MATRIX
# ----------------------------
print("[6] Computing similarity matrix...")

text_sim = cosine_similarity(X_text)

print("Similarity matrix computed")

# ----------------------------
# INDEX MAPPING
# ----------------------------
print("[7] Building index mapping...")

indices = pd.Series(
    df_food_business.index,
    index=df_food_business["business_id"]
)

# ----------------------------
# SAVE ARTIFACTS
# ----------------------------
print("[8] Saving artifacts...")

# dataframe
df_food_business.to_parquet("models/df_food_business.parquet", index=False)

# embeddings
np.save("models/X_text.npy", X_text)

# similarity matrix
np.save("models/text_sim.npy", text_sim)

# index mapping
indices.to_pickle("models/indices.pkl")


print("All artifacts saved")

# ----------------------------
# END
# ----------------------------
print("=== DONE ===")