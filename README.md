<p align="center">
  <img src="assets/Where_to_eat.png" width="100%" />
</p>

**Where to Eat** is an AI-powered restaurant recommendation application that helps users discover **food businesses (restaurants, cafés, bars, and similar places)** based on categories or free-text descriptions.

---

## ✨ Features

* Explore restaurants by selecting **what you like** (categories, atmosphere, extras)
* Filter by **price range** (Easy on the wallet / Comfortable / Treat yourself / Premium experience)
* Filter by **city** for location-aware recommendations
* View detailed restaurant **attributes** (WiFi, parking, noise level, dress code, etc.) with one click
* Personalized recommendations using a **like-based feedback system**
* "More like this" suggestions after each interaction
* Clean and interactive Streamlit interface

---

## 🧠 How it works (simple explanation)

The system uses a **transformer-based AI model** that understands **meaning, not just keywords**.

Whether you type a free-text query like *"romantic rooftop dinner with jazz"* or select categories like *"sushi" + "trendy"*, the model captures the semantic intent behind your request and finds restaurants that truly match your mood.

This means:
- You can describe experiences, not just dish names
- Your query is understood in context, not just matched by word presence
- Category selection works the same way — the AI understands what "casual", "romantic", or "hipster" really means

The system converts everything (restaurant descriptions + user inputs) into numerical representations called **embeddings**, then compares them to find the most relevant matches.

---

## 🚀 How to run the app

This project uses **uv** for dependency and environment management.

### 1. Install dependencies

```bash
uv sync
```

### 2. Launch the Streamlit app

```bash
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
```