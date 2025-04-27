import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model 
import pickle



st.set_page_config(
    page_title="Sentiment Analyzer",  # Title on browser tab
   #page_icon="🧠",                   # Favicon (emoji or upload custom)
    layout="wide",                    # 'centered' or 'wide'
    initial_sidebar_state="expanded", # 'auto', 'expanded', 'collapsed'
)

# Load vectorizer
with open('st/model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load model
model = load_model('st/model/sentiment_model.h5')


#page title and description
st.title("Sentiment Predictor")
st.write("Enter a sentence and predict its sentiment (Positive, Negative, Neutral)")
st.write("-----")

# Text input from user
st.header("🧠 Sentiment Predictor")
user_input = st.text_area("Enter a review to analyze:")


if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize the user input
        sample_vec = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(sample_vec)
        predicted_class = prediction.argmax(axis=-1)[0]

        # Label map
        label_map = {0: "Negative", 1: "Somewhat Negative", 2: "Neutral", 3: "Somewhat Positive", 4: "Positive"}

        # Output
        st.success(f"🧠 Sentiment Prediction: {label_map.get(predicted_class, 'Unknown')}")
        
st.write("-----")
st.warning("Note: The model is trained on a specific dataset and may not generalize well to all types of text. Use with caution.")