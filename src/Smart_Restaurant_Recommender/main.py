import streamlit as st
import numpy as np

from Yelp_recommender.preprocessing import load_preprocessed
from Yelp_recommender.recommender import get_reco
from Yelp_recommender.utils import (
    extract_categories,
    build_user_embedding,
    get_initial_recos,
    encode_text
)

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

/* ── ROOT ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── REMOVE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem; max-width: 1200px; }

/* ── ANIMATED GRADIENT BANNER ── */
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

/* ── PILL BADGE ── */
.pill {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-right: 6px;
}

/* ── MODE RADIO ── */
div[data-testid="stRadio"] > label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--muted) !important;
    font-weight: 500;
}
div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 12px !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 10px 22px !important;
    transition: all 0.2s ease;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    border-color: var(--coral);
    background: #1f1010;
}

/* ── SECTION LABEL ── */
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 28px 0 16px 0;
    color: var(--text);
}

/* ── DIVIDER ── */
.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--coral), var(--gold), var(--mint), var(--sky), var(--plum));
    border-radius: 2px;
    margin: 28px 0;
    border: none;
}

/* ── MULTISELECT ── */
div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: linear-gradient(135deg, var(--coral), var(--gold)) !important;
    color: #000 !important;
    font-weight: 600 !important;
    border-radius: 999px !important;
    font-size: 12px !important;
}
div[data-testid="stMultiSelect"] > div {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── TEXT INPUT ── */
div[data-testid="stTextInput"] input {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 1rem !important;
    padding: 14px 18px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s ease;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--coral) !important;
    box-shadow: 0 0 0 3px rgba(255,107,107,0.15) !important;
}

/* ── MAIN BUTTON ── */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--coral) 0%, var(--gold) 100%) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 20px rgba(255,107,107,0.35) !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,107,107,0.45) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── RESULTS TITLE ── */
.results-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(90deg, var(--coral), var(--gold));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 36px 0 20px 0;
}

/* ── BUSINESS CARD ── */
.biz-card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 6px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.biz-card::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: 4px 0 0 4px;
}
.biz-card:nth-child(1)::before  { background: var(--coral); }
.biz-card:nth-child(2)::before  { background: var(--gold); }
.biz-card:nth-child(3)::before  { background: var(--mint); }
.biz-card:nth-child(4)::before  { background: var(--sky); }
.biz-card:nth-child(5)::before  { background: var(--plum); }
.biz-card:nth-child(6)::before  { background: var(--coral); }
.biz-card:nth-child(7)::before  { background: var(--gold); }
.biz-card:nth-child(8)::before  { background: var(--mint); }
.biz-card:nth-child(9)::before  { background: var(--sky); }
.biz-card:nth-child(10)::before { background: var(--plum); }
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

/* ── SIMILAR CARD ── */
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

/* ── EXPANDER ── */
div[data-testid="stExpander"] {
    background: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 14px !important;
    margin-bottom: 6px;
}
div[data-testid="stExpander"] summary {
    color: var(--plum) !important;
    font-weight: 600;
    font-size: 0.9rem;
}

/* ── WARNING ── */
div[data-testid="stAlert"] {
    background: #1f1a0e !important;
    border: 1px solid var(--gold) !important;
    border-radius: 12px !important;
    color: var(--gold) !important;
}

/* ── LIKE / DISLIKE BUTTON OVERRIDE ── */
.like-btn button {
    background: linear-gradient(135deg, #1a3a1a, #0f2b0f) !important;
    border: 1.5px solid var(--mint) !important;
    color: var(--mint) !important;
    font-size: 0.88rem !important;
    padding: 8px 20px !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    width: 100% !important;
}
.dislike-btn button {
    background: linear-gradient(135deg, #2a1a1a, #1f0f0f) !important;
    border: 1.5px solid #555 !important;
    color: #888 !important;
    font-size: 0.88rem !important;
    padding: 8px 20px !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    width: 100% !important;
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
ALL_CATEGORIES = extract_categories(df_food_business)

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
selected_categories = None

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ----------------------------
# INPUT ZONE
# ----------------------------
if mode == "🎯 Choose what you like":
    st.markdown('<p class="section-label">Choose your mood</p>', unsafe_allow_html=True)
    selected_categories = st.multiselect(
        "Select categories that match your moment",
        ALL_CATEGORIES,
        max_selections=5,
        label_visibility="collapsed"
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
ACCENT_COLORS = ["#FF6B6B", "#FFD93D", "#6BCB77", "#4ECDC4", "#C77DFF",
                 "#FF6B6B", "#FFD93D", "#6BCB77", "#4ECDC4", "#C77DFF"]

def display_business_card(row, card_class="biz-card", name_class="biz-name",
                           desc_class="biz-desc", addr_class="biz-addr", accent_color="#FF6B6B"):
    st.markdown(
        f"""
        <div class="{card_class}" style="border-left: 4px solid {accent_color};">
            <p class="{name_class}">🍽️ {row.get('name', 'N/A')}</p>
            <p class="{desc_class}">{row.get('description', '')}</p>
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
        if not selected_categories:
            st.warning("Pick at least one categorie to explore.")
        else:
            user_emb = build_user_embedding(selected_categories)
            st.session_state.initial_recos = get_initial_recos(
                user_emb, X_text, df_food_business, top_n=10
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

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="like-btn">', unsafe_allow_html=True)
            if st.button("👍 Like", key=f"like_{idx}"):
                st.session_state.liked_recos[row["business_id"]] = get_reco(
                    row["business_id"], top_n=5
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="dislike-btn">', unsafe_allow_html=True)
            st.button("👎 Not for me", key=f"dis_{idx}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Similar places
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