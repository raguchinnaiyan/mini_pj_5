import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import os
import torch
import streamlit as st
from transformers import pipeline

# Streamlit UI
st.title("Sentiment Predictor")
st.write("Enter a sentence below to analyze its sentiment.")


# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", device=-1)  # -1 means use CPU


# Text input
user_text = st.text_input("Enter a sentence:")

# Button to trigger prediction
if st.button("Predict Sentiment"):
    if user_text.strip() != "":
        # Perform sentiment analysis
        sentiment = sentiment_analyzer(user_text)
        label = sentiment[0]['label']
        score = sentiment[0]['score']
        
        # Display result
        st.subheader("Sentiment Analysis Result:")
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence Score:** {score:.2f}")
    else:
        st.warning("Please enter a sentence to analyze.")
        
st.write("-----")
st.warning("Note: The model is trained on a specific dataset and may not generalize well to all types of text. Use with caution.")