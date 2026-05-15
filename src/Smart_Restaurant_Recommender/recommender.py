from pathlib import Path

import numpy as np
import pandas as pd

from Smart_Restaurant_Recommender.preprocessing import load_preprocessed
from Smart_Restaurant_Recommender.utils import haversine

# ----------------------------
# PATHS (robust)
df_food_business, X_text, text_sim, indices = load_preprocessed()

# rebuild index mapping after reset
indices = pd.Series(df_food_business.index, index=df_food_business["business_id"])

# ----------------------------
# RECOMMENDER FUNCTION
# ----------------------------


def get_reco(
    business_id,
    top_n=5,
    city_threshold=15,
    alpha=0.7,  # weight for text similarity
    beta=0.3,  # weight for geographic proximity
):
    # Get index of the reference business
    idx = indices[business_id]

    # Text similarity scores (cosine similarity from embedding matrix)
    text_scores = text_sim[idx].copy()

    # Coordinates of the reference business
    lat1 = df_food_business.iloc[idx]["latitude"]
    lon1 = df_food_business.iloc[idx]["longitude"]

    # Coordinates of all businesses
    lat2 = df_food_business["latitude"].values
    lon2 = df_food_business["longitude"].values

    # Compute geographic distances (Haversine)
    dists = haversine(lat1, lon1, lat2, lon2)

    # Proximity score normalized between 0 and 1 (closer = higher score)
    prox_scores = 1 / (1 + dists)
    prox_scores = prox_scores / prox_scores.max()

    # Final combined score (text similarity + proximity)
    final_scores = alpha * text_scores + beta * prox_scores

    # Strong penalty for businesses outside the defined city radius
    outside_mask = dists > city_threshold
    final_scores[outside_mask] *= 0.5

    # Exclude the reference business itself
    final_scores[idx] = -1

    # Get top-N recommendations
    top_idx = np.argsort(final_scores)[::-1][:top_n]

    return list(df_food_business.iloc[top_idx]["business_id"].unique())
