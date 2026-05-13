import numpy as np
from Yelp_recommender.preprocessing import load_preprocessed
from Yelp_recommender.utils import haversine

df_food_business, X_text, text_sim, indices = load_preprocessed()


def get_reco(business_id, sim_matrix=None, top_n=5, city_threshold=15):

    if sim_matrix is None:
        sim_matrix = text_sim

    idx = indices[business_id]

    sims = sim_matrix[idx]

    lat1 = df_food_business.iloc[idx]["latitude"]
    lon1 = df_food_business.iloc[idx]["longitude"]

    lat2 = df_food_business["latitude"].values
    lon2 = df_food_business["longitude"].values

    dists = haversine(lat1, lon1, lat2, lon2)

    same_city_idx = np.where(dists <= city_threshold)[0]
    outside_idx = np.where(dists > city_threshold)[0]

    same_sorted = same_city_idx[np.argsort(sims[same_city_idx])[::-1]]
    outside_sorted = outside_idx[np.argsort(sims[outside_idx])[::-1]]

    final_idx = np.concatenate([same_sorted, outside_sorted])
    final_idx = final_idx[final_idx != idx]

    recos = df_food_business.iloc[final_idx[:top_n]]

    return recos["business_id"].tolist()