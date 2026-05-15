import streamlit as st
import numpy as np
import pandas as pd

from Smart_Restaurant_Recommender.preprocessing import load_preprocessed, DATA_DIR
from Smart_Restaurant_Recommender.recommender import get_reco
from Smart_Restaurant_Recommender.utils import (
    extract_categories,
    build_user_embedding,
    get_initial_recos,
    encode_text
)

from Smart_Restaurant_Recommender.preprocessing import DATA_DIR

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Where To Eat", layout="wide", page_icon="🍽️")

# ----------------------------
# GLOBAL CSS
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --coral:    #FF6B6B;
    --gold:     #FFD93D;
    --mint:     #6BCB77;
    --sky:      #4ECDC4;
    --plum:     #C77DFF;
    --bg:       #0D0D0D;
    --surface:  #161616;
    --card:     #1E1E1E;
    --border:   #2A2A2A;
    --text:     #F0EDE8;
    --muted:    #888;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem; max-width: 1200px; }

.hero-banner {
    background: linear-gradient(135deg, #FF6B6B 0%, #FFD93D 25%, #6BCB77 50%, #4ECDC4 75%, #C77DFF 100%);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 24px;
    padding: 56px 48px;
    margin: 24px 0 36px 0;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(0,0,0,0.42);
    border-radius: 24px;
}
.hero-banner > * { position: relative; z-index: 1; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    color: #fff;
    margin: 0 0 10px 0;
    line-height: 1.1;
    letter-spacing: -1px;
}

.hero-sub {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.82);
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.5px;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--coral), var(--gold), var(--mint), var(--sky), var(--plum));
    border-radius: 2px;
    margin: 28px 0;
    border: none;
}

.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 28px 0 16px 0;
    color: var(--text);
}

.results-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(90deg, var(--coral), var(--gold));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 36px 0 20px 0;
}

.biz-card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 6px;
}

.biz-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 6px 0;
}

.biz-desc {
    font-size: 0.9rem;
    color: #bbb;
    margin: 0 0 10px 0;
    line-height: 1.6;
}

.biz-addr {
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.3px;
}

.sim-card {
    background: #252525;
    border: 1px solid #333;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 8px;
}

.sim-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 4px 0;
}

.sim-desc {
    font-size: 0.82rem;
    color: #aaa;
    margin: 0 0 6px 0;
}

.sim-addr {
    font-size: 0.72rem;
    color: var(--muted);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🍽️ Where To Eat</p>
    <p class="hero-sub">Discover places that match your mood, not just your keywords.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df_food_business, X_text, _, _ = load_preprocessed()
df_food_business = df_food_business.reset_index(drop=True)

df_mic_mac = pd.read_parquet(DATA_DIR / "micro_to_macro.parquet")

food_cats = df_mic_mac[df_mic_mac["macro_category"] == "Food & Beverage"]["micro_category"].unique()
extra_cats = df_mic_mac[df_mic_mac["macro_category"] != "Food & Beverage"]["micro_category"].unique()

ALL_CATEGORIES = extract_categories(df_food_business)

# ----------------------------
# ATMOSPHERE OPTIONS
# ----------------------------
ALL_ATMOSPHERES = [
    "casual",
    "trendy",
    "romantic",
    "upscale",
    "classy",
    "hipster",
    "divey",
    "intimate",
    "touristy",
]

# ----------------------------
# SESSION STATE
# ----------------------------
if "initial_recos" not in st.session_state:
    st.session_state.initial_recos = None

if "liked_recos" not in st.session_state:
    st.session_state.liked_recos = {}

# ----------------------------
# MODE
# ----------------------------
mode = st.radio(
    "HOW DO YOU WANT TO EXPLORE?",
    ["🎯 Choose what you like", "✍️ Tell us what you want"],
    horizontal=True
)

user_query = None
selected_categories = []
selected_extras = []
selected_atmospheres = []

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ----------------------------
# INPUT ZONE
# ----------------------------
if mode == "🎯 Choose what you like":
    st.markdown('<p class="section-label">Choose your mood</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_categories = st.multiselect(
            "🍽️ What you usually enjoy eating",
            food_cats,
            placeholder="e.g. sushi, burgers, italian food",
            max_selections=5
        )

    with col2:
        st.markdown("### ➕ Extras")

        selected_extras = st.multiselect(
            "",
            sorted(extra_cats),
            placeholder="Cinema, shopping, wellness..."
        )

    with col3:
        selected_atmospheres = st.multiselect(
            "✨ What kind of vibe you're looking for",
            ALL_ATMOSPHERES,
            placeholder="e.g. romantic, casual, trendy"
        )

else:
    st.markdown('<p class="section-label">Describe your plan</p>', unsafe_allow_html=True)
    user_query = st.text_input(
        "mood",
        placeholder="e.g. romantic rooftop dinner, loud night out with music, chill brunch with friends",
        label_visibility="collapsed"
    )

# ----------------------------
# BUTTON
# ----------------------------
st.markdown("<br>", unsafe_allow_html=True)
generate = st.button("✨ Find places for me")

# ----------------------------
# CARD DISPLAY
# ----------------------------
ACCENT_COLORS = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4ECDC4", "#C77DFF",
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4ECDC4", "#C77DFF"
]

def display_business_card(
    row,
    card_class="biz-card",
    name_class="biz-name",
    desc_class="biz-desc",
    addr_class="biz-addr",
    accent_color="#FF6B6B"
):
    st.markdown(
        f"""
        <div class="{card_class}" style="border-left: 4px solid {accent_color};">
            <p class="{name_class}">🍽️ {row.get('name', 'N/A')}</p>
            <p class="{desc_class}">{row.get('full_desc', '')}</p>
            <p class="{addr_class}">📍 {row.get('address', '')}, {row.get('city', '')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# RECOMMENDATION ENGINE
# ----------------------------
if generate:
    if mode == "🎯 Choose what you like":
        if not selected_categories and not selected_atmospheres and not selected_extras:
            st.warning("Pick at least one option.")
        else:
            user_emb = build_user_embedding(
                selected_categories,
                selected_extras,
                selected_atmospheres
            )

            st.session_state.initial_recos = get_initial_recos(
                user_emb,
                X_text,
                df_food_business,
                top_n=10
            )

    else:
        if not user_query:
            st.warning("Tell us what you want first.")
        else:
            user_emb = encode_text(user_query)
            scores = np.dot(X_text, user_emb)
            top_idx = np.argsort(scores)[::-1][:10]
            st.session_state.initial_recos = df_food_business.iloc[top_idx]

# ----------------------------
# RESULTS
# ----------------------------
if st.session_state.initial_recos is not None:

    st.markdown('<p class="results-title">🔥 Your matches</p>', unsafe_allow_html=True)

    for i, (idx, row) in enumerate(st.session_state.initial_recos.iterrows()):
        accent = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        display_business_card(row, accent_color=accent)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Like", key=f"like_{idx}"):
                st.session_state.liked_recos[row["business_id"]] = get_reco(
                    row["business_id"],
                    top_n=5
                )

        with col2:
            st.button("👎 Not for me", key=f"dis_{idx}")

        if row["business_id"] in st.session_state.liked_recos:
            with st.expander("✨ Because you liked this", expanded=True):
                recos_ids = st.session_state.liked_recos[row["business_id"]]

                similar_df = df_food_business[
                    df_food_business["business_id"].isin(recos_ids)
                ]

                cols = st.columns(len(similar_df))

                for j, (_, rec_row) in enumerate(similar_df.iterrows()):
                    with cols[j]:
                        display_business_card(
                            rec_row,
                            card_class="sim-card",
                            name_class="sim-name",
                            desc_class="sim-desc",
                            addr_class="sim-addr",
                            accent_color=ACCENT_COLORS[j % len(ACCENT_COLORS)]
                        )

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)