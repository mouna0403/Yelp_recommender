import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from Yelp_recommender.preprocessing import run_preprocessing
from Yelp_recommender.recommender import get_reco
from Yelp_recommender.utils import (
    extract_categories,
    build_user_embedding,
    get_initial_recos,
)

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

# ----------------------------
# PREPROCESSING (RUN ONCE PER SESSION)
# ----------------------------
if "data_ready" not in st.session_state:

    # run only if files do not exist
    required_files = [
        MODEL_DIR / "df_food_business.parquet",
        MODEL_DIR / "X_text.npy",
        MODEL_DIR / "text_sim.npy",
        MODEL_DIR / "indices.pkl",
    ]

    if not all(f.exists() for f in required_files):
        run_preprocessing()

    st.session_state.data_ready = True

# ----------------------------
# LOAD DATA (SAFE AFTER PREPROCESSING)
# ----------------------------
df_food_business = pd.read_parquet(MODEL_DIR / "df_food_business.parquet")
X_text = np.load(MODEL_DIR / "X_text.npy")

df_food_business = df_food_business.reset_index(drop=True)

ALL_CATEGORIES = extract_categories(df_food_business)

# ----------------------------
# SESSION STATE
# ----------------------------
if "initial_recos" not in st.session_state:
    st.session_state.initial_recos = None

if "liked_recos" not in st.session_state:
    st.session_state.liked_recos = {}

# ----------------------------
# DISPLAY FUNCTION
# ----------------------------
def display_business_card(row):
    st.markdown(f"**{row.get('name', 'N/A')}**")
    st.write(row.get("description", ""))
    st.caption(f"{row.get('address', '')}, {row.get('city', '')}")

# ----------------------------
# UI
# ----------------------------
st.title("Food Recommendation System")

selected_categories = st.multiselect(
    "Select categories (max 10)",
    ALL_CATEGORIES,
    max_selections=10
)

# ----------------------------
# GENERATE INITIAL RECOMMENDATIONS
# ----------------------------
if st.button("Generate recommendations"):

    if len(selected_categories) == 0:
        st.warning("Select at least one category.")
    else:
        user_emb = build_user_embedding(selected_categories)

        st.session_state.initial_recos = get_initial_recos(
            user_emb,
            X_text,
            df_food_business,
            top_n=10
        )

# ----------------------------
# DISPLAY INITIAL RECOMMENDATIONS
# ----------------------------
if st.session_state.initial_recos is not None:

    st.subheader("Initial recommendations")

    for idx, row in st.session_state.initial_recos.iterrows():

        display_business_card(row)

        col1, col2 = st.columns(2)

        # ---------------- LIKE ----------------
        with col1:
            if st.button("Like", key=f"like_{idx}"):

                st.session_state.liked_recos[row["business_id"]] = get_reco(
                    row["business_id"],
                    top_n=5
                )

        # ---------------- DISLIKE ----------------
        with col2:
            st.button("Dislike", key=f"dis_{idx}")

        # ---------------- SIMILAR ITEMS ----------------
        if row["business_id"] in st.session_state.liked_recos:

            recos_ids = st.session_state.liked_recos[row["business_id"]]

            similar_df = df_food_business[
                df_food_business["business_id"].isin(recos_ids)
            ]

            st.markdown("**Recommended based on this**")

            cols = st.columns(len(similar_df))

            for i, (_, rec_row) in enumerate(similar_df.iterrows()):
                with cols[i]:
                    display_business_card(rec_row)

        st.divider()