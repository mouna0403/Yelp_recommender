# 🍽️ Food Explorer

Food Explorer is a restaurant recommendation application that helps users discover **food businesses (restaurants, cafés, and similar places)** based on mood, vibe, or free-text descriptions.

---

## ✨ Features

* Explore restaurants by **vibe / categories** (e.g. brunch, sushi, nightlife)
* Search using **free-text input** (e.g. “romantic dinner”, “chill rooftop bar”)
* Personalized recommendations using a **like-based feedback system**
* “More like this” suggestions after each interaction
* Clean and interactive Streamlit interface

---

## 🧠 How it works (simple explanation)

The system uses a **transformer-based text model** to understand both:

* restaurant descriptions
* user queries

It converts text into embeddings (numerical representations) and compares them to find the most relevant matches.

---

## 📊 Data exploration

All the data exploration, analysis, and reasoning behind the approach is available in the `notebooks/` folder.

This helps understand:

* how the dataset was cleaned
* how food businesses were filtered
* how categories and text data were processed
* the intuition behind the recommendation strategy

---

## 🚀 How to run the app

This project uses **uv** for dependency and environment management.

### 1. Install dependencies

```bash id="uv1"
uv sync
```

### 2. Launch the Streamlit app

```bash id="uv2"
uv run streamlit run src/Yelp_recommender/main.py
```

---

## ⚙️ Requirements

* Python 3.10+
* uv installed
* dataset included in the project structure

---

## 📌 Deployment note

A Docker configuration is not provided at the moment, as the current pipeline includes heavy preprocessing and evolving dependencies. Containerization may be added in a future iteration once the architecture is stabilized.

---

## 🔮 Future improvements

* deeper personalization based on user history
* performance optimization of embeddings and similarity search
* production-ready deployment setup
