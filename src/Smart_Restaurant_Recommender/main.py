import streamlit as st
import numpy as np
import pandas as pd

from Smart_Restaurant_Recommender.preprocessing import load_preprocessed, DATA_DIR
from Smart_Restaurant_Recommender.recommender import get_reco
from Smart_Restaurant_Recommender.utils import (
    extract_categories,
    build_user_embedding,
    get_initial_recos,
    encode_text,
)

st.set_page_config(page_title="Where To Eat", layout="wide", page_icon="🍽️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,900;1,900&family=Plus+Jakarta+Sans:wght@300;400;700&display=swap');

:root {
    --coral:   #FF6B6B;
    --gold:    #FFD93D;
    --mint:    #6BCB77;
    --sky:     #4ECDC4;
    --plum:    #C77DFF;
    --bg:      #0D0D0D;
    --surface: #161616;
    --card:    #1E1E1E;
    --border:  #2A2A2A;
    --text:    #F0EDE8;
    --muted:   #888;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem; max-width: 1200px; }

.stApp::before {
    content: "";
    position: fixed;
    top: -100px;
    left: -100px;
    width: 450px;
    height: 450px;
    background: radial-gradient(circle, rgba(255,107,107,0.12) 0%, rgba(0,0,0,0) 70%);
    z-index: -1;
    pointer-events: none;
}

.custom-header {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 10px 0 30px 0;
}
.chef-logo {
    background: linear-gradient(135deg, #FF6B6B, #FFD93D);
    width: 65px;
    height: 65px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
    box-shadow: 0 10px 20px rgba(255,107,107,0.25);
    transform: rotate(-4deg);
    transition: transform 0.3s ease;
}
.chef-logo:hover { transform: rotate(0deg) scale(1.05); }
.brand-text h1 {
    font-family: 'Playfair Display', serif;
    color: white;
    margin: 0;
    font-size: 26px;
    letter-spacing: -0.5px;
    font-weight: 900;
}
.brand-text p {
    color: var(--coral);
    font-size: 10px;
    letter-spacing: 3px;
    font-weight: 700;
    margin: 0;
    text-transform: uppercase;
}

.hero-box {
    background: var(--surface);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 35px;
    padding: 50px;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
}
.hero-box h2 {
    font-family: 'Playfair Display', serif;
    font-size: 48px !important;
    font-weight: 900;
    color: white;
    margin-bottom: 5px;
    line-height: 1.2;
}
.hero-box span { color: var(--coral); }
.hero-box p { color: #777; font-weight: 300; }

.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--coral), var(--gold), var(--mint), var(--sky), var(--plum));
    border-radius: 2px;
    margin: 28px 0;
    border: none;
}

.stRadio > div {
    background: rgba(255,255,255,0.03);
    padding: 15px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.05);
}

.stButton>button {
    border-radius: 18px !important;
    border: none !important;
    padding: 12px 30px !important;
    font-weight: 700 !important;
    background: #F0EDE8 !important;
    color: #0D0D0D !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
.stButton>button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 10px 20px rgba(240,237,232,0.15) !important;
}

.biz-card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 6px;
    transition: background 0.3s ease, transform 0.3s ease;
}
.biz-card:hover {
    background: #242424;
    transform: translateX(4px);
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
.sim-desc { font-size: 0.82rem; color: #aaa; margin: 0 0 6px 0; }
.sim-addr { font-size: 0.72rem; color: var(--muted); }

.results-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(90deg, var(--coral), var(--gold));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 36px 0 20px 0;
}

/* Slide-up animation for main cards */
@keyframes slideInUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Scale-in animation for sim cards */
@keyframes fadeInScale {
    from { opacity: 0; transform: scale(0.94); }
    to   { opacity: 1; transform: scale(1); }
}

.card-animate {
    animation: slideInUp 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.sim-animate {
    animation: fadeInScale 0.35s cubic-bezier(0.22, 1, 0.36, 1) both;
}

/* Searching banner */
@keyframes searchPulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(255,107,107,0); }
    50%       { opacity: 0.7; box-shadow: 0 0 30px 6px rgba(255,107,107,0.1); }
}
@keyframes dotBounce {
    0%, 80%, 100% { transform: translateY(0); }
    40%           { transform: translateY(-7px); }
}
.searching-banner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    background: linear-gradient(135deg, rgba(255,107,107,0.10), rgba(255,217,61,0.06));
    border: 1px solid rgba(255,107,107,0.22);
    border-radius: 20px;
    padding: 30px 40px;
    margin: 24px 0;
    animation: searchPulse 1.8s ease-in-out infinite;
}
.searching-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: var(--coral);
    letter-spacing: 1px;
}
.dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    animation: dotBounce 1.2s ease-in-out infinite;
}
.dot-1 { background: var(--coral); animation-delay: 0s; }
.dot-2 { background: var(--gold);  animation-delay: 0.18s; }
.dot-3 { background: var(--mint);  animation-delay: 0.36s; }

.footer {
    text-align: center;
    color: #333;
    font-size: 9px;
    margin-top: 80px;
    letter-spacing: 4px;
    font-weight: 700;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-header">
    <div class="chef-logo">👨‍🍳</div>
    <div class="brand-text">
        <h1>WHERE TO EAT</h1>
        <p>Smart Recommender</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <h2>Discover where to <span>Eat</span></h2>
    <p>Find places that match your mood, not just your keywords.</p>
</div>
""", unsafe_allow_html=True)

df_food_business, X_text, _, _ = load_preprocessed()
df_food_business = df_food_business.reset_index(drop=True)

df_mic_mac = pd.read_parquet(DATA_DIR / "micro_to_macro.parquet")
food_cats  = df_mic_mac[df_mic_mac["macro_category"] == "Food & Beverage"]["micro_category"].unique()
extra_cats = df_mic_mac[df_mic_mac["macro_category"] != "Food & Beverage"]["micro_category"].unique()

ALL_CATEGORIES  = extract_categories(df_food_business)
ALL_ATMOSPHERES = [
    "casual", "trendy", "romantic", "upscale",
    "classy", "hipster", "divey", "intimate", "touristy",
]

if "initial_recos" not in st.session_state:
    st.session_state.initial_recos = None
if "liked_recos" not in st.session_state:
    st.session_state.liked_recos = {}

mode = st.radio(
    "HOW DO YOU WANT TO EXPLORE?",
    ["🎯 Choose what you like", "✍️ Tell us what you want"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

user_query           = None
selected_categories  = []
selected_extras      = []
selected_atmospheres = []

if mode == "🎯 Choose what you like":
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_categories = st.multiselect(
            "🍽️ Categories", food_cats,
            placeholder="Choose food", max_selections=5,
        )
    with col2:
        selected_atmospheres = st.multiselect(
            "✨ Atmosphere", ALL_ATMOSPHERES,
            placeholder="Choose a vibe",
        )
    with col3:
        selected_extras = st.multiselect(
            "➕ Extras", sorted(extra_cats),
            placeholder="Cinema, shopping…",
        )
else:
    st.text_input(
        "mood",
        placeholder="e.g. romantic rooftop dinner with jazz music...",
        label_visibility="collapsed",
        key="text_query",
    )
    user_query = st.session_state.get("text_query", "")

st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
generate = st.button("✨ FIND PLACES FOR ME")

ACCENT_COLORS = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4ECDC4", "#C77DFF",
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4ECDC4", "#C77DFF",
]


def display_business_card(row, card_class="biz-card", name_class="biz-name",
                           desc_class="biz-desc", addr_class="biz-addr",
                           accent_color="#FF6B6B", delay=0.0):
    anim_class = "sim-animate" if card_class == "sim-card" else "card-animate"
    st.markdown(
        f"""
        <div class="{card_class} {anim_class}"
             style="border-left: 4px solid {accent_color}; animation-delay: {delay:.2f}s;">
            <p class="{name_class}">🍽️ {row.get('name', 'N/A')}</p>
            <p class="{desc_class}">{row.get('full_desc', '')}</p>
            <p class="{addr_class}">📍 {row.get('address', '')}, {row.get('city', '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if generate:
    valid = False

    if mode == "🎯 Choose what you like":
        if not selected_categories and not selected_atmospheres and not selected_extras:
            st.warning("Pick at least one option.")
        else:
            valid = True
    else:
        if not user_query:
            st.warning("Tell us what you want first.")
        else:
            valid = True

    if valid:
        searching_slot = st.empty()
        searching_slot.markdown("""
        <div class="searching-banner">
            <span class="searching-text">Searching</span>
            <span class="dot dot-1"></span>
            <span class="dot dot-2"></span>
            <span class="dot dot-3"></span>
        </div>
        """, unsafe_allow_html=True)

        if mode == "🎯 Choose what you like":
            user_emb = build_user_embedding(selected_categories, selected_extras, selected_atmospheres)
            st.session_state.initial_recos = get_initial_recos(user_emb, X_text, df_food_business, top_n=10)
        else:
            user_emb = encode_text(user_query)
            scores   = np.dot(X_text, user_emb)
            top_idx  = np.argsort(scores)[::-1][:10]
            st.session_state.initial_recos = df_food_business.iloc[top_idx]

        searching_slot.empty()


if st.session_state.initial_recos is not None:

    st.markdown('<p class="results-title">🔥 Your matches</p>', unsafe_allow_html=True)

    for i, (idx, row) in enumerate(st.session_state.initial_recos.iterrows()):
        accent = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        display_business_card(row, accent_color=accent, delay=i * 0.07)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Like", key=f"like_{idx}"):
                st.session_state.liked_recos[row["business_id"]] = get_reco(
                    row["business_id"], top_n=5
                )
        with col2:
            st.button("👎 Not for me", key=f"dis_{idx}")

        if row["business_id"] in st.session_state.liked_recos:
            with st.expander("✨ Because you liked this", expanded=True):
                recos_ids  = st.session_state.liked_recos[row["business_id"]]
                similar_df = df_food_business[df_food_business["business_id"].isin(recos_ids)]
                cols       = st.columns(len(similar_df))
                for j, (_, rec_row) in enumerate(similar_df.iterrows()):
                    with cols[j]:
                        display_business_card(
                            rec_row,
                            card_class="sim-card",
                            name_class="sim-name",
                            desc_class="sim-desc",
                            addr_class="sim-addr",
                            accent_color=ACCENT_COLORS[j % len(ACCENT_COLORS)],
                            delay=j * 0.08,
                        )

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">© SMART RESTAURANT RECOMMENDER — POWERED BY AI</div>
""", unsafe_allow_html=True)