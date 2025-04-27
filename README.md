# mini_pj_5
A dataset of ChatGPT reviews with labeled sentiment (positive, negative, neutral) for sentiment analysis. Includes Streamlit setup for easy deployment of a sentiment analysis app.

This project performs sentiment analysis on ChatGPT reviews using a pre-trained machine learning model. It classifies reviews as Positive, Negative, or Neutral. Built with Streamlit, the app allows users to input reviews and get predictions in real-time. The project includes a visualization page to explore sentiment distribution in the dataset.

This project uses Raw_Data (a dataset of ChatGPT reviews) to train a sentiment analysis model. The VS file (Jupyter notebook) handles model training. The Streamlit UI (in the st/ folder) serves as the frontend for real-time sentiment classification of user reviews. The app predicts whether reviews are Positive, Negative, or Neutral and visualizes sentiment distribution.

``
mini_pj_5/
│
├── st/                                # 📦 Main app folder
│   ├── 1_Predictor.py                 # 🏠 Main page for sentiment analysis
│
│   ├── models/                        # 🤖 Pre-trained ML models
│   │   ├── sentiment_model.h5         # - Model for sentiment analysis
│   │   └── tfidf_vectorizer.pkl       # - TF-IDF vectorizer used for text processing
│
│   └── pages/                         # 📄 Sub-pages (auto-loaded by Streamlit)
│       ├── 2_Visualization.py         # - Page for visualization and insights
│
├── .streamlit/                        # Streamlit configuration
│   └── config.toml                    # Streamlit config settings
│
├── README.md                          # 📘 Project description and setup guide
