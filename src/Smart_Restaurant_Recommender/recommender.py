import numpy as np
import pandas as pd
from pathlib import Path

from Yelp_recommender.utils import haversine

# ----------------------------
# PATHS (robust)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

# ----------------------------
# LOAD ARTIFACTS
# ----------------------------
df_food_business = pd.read_parquet(MODEL_DIR / "df_food_business.parquet")
X_text = np.load(MODEL_DIR / "X_text.npy")
text_sim = np.load(MODEL_DIR / "text_sim.npy")
indices = pd.read_pickle(MODEL_DIR / "indices.pkl")

df_food_business = df_food_business.reset_index(drop=True)

# rebuild index mapping after reset
indices = pd.Series(
    df_food_business.index,
    index=df_food_business["business_id"]
)

# ----------------------------
# RECOMMENDER FUNCTION
# ----------------------------
def get_reco(business_id, sim_matrix=text_sim, top_n=5, city_threshold=15):
    """
    Recommends similar businesses using text similarity and geographic proximity.
    """

    idx = indices[business_id]

    # similarity scores for target item
    sims = sim_matrix[idx]

    # target coordinates
    lat1 = df_food_business.iloc[idx]["latitude"]
    lon1 = df_food_business.iloc[idx]["longitude"]

    # all coordinates
    lat2 = df_food_business["latitude"].values
    lon2 = df_food_business["longitude"].values

    # geographic distances
    dists = haversine(lat1, lon1, lat2, lon2)

    same_city_mask = dists <= city_threshold

    same_city_idx = np.where(same_city_mask)[0]
    outside_idx = np.where(~same_city_mask)[0]

    # sort by similarity score
    same_sorted = same_city_idx[np.argsort(sims[same_city_idx])[::-1]]
    outside_sorted = outside_idx[np.argsort(sims[outside_idx])[::-1]]

    # combine results (same city prioritized)
    final_idx = np.concatenate([same_sorted, outside_sorted])

    # remove self index
    final_idx = final_idx[final_idx != idx]

    recos = df_food_business.iloc[final_idx[:top_n]]

    return recos["business_id"].tolist()


# ----------------------------
# OPTIONAL LOCAL TEST
# ----------------------------
if __name__ == "__main__":
    sample_id = df_food_business["business_id"].iloc[0]
    print(get_reco(sample_id))