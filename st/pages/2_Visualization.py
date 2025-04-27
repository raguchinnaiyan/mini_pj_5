import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

#page title and description
st.title("Sentiment Analysis Visualization")
st.markdown("This is a sentiment analysis visualization dashboard for ChatGPT reviews. You can explore various aspects of the reviews, including overall sentiment distribution, sentiment by rating, keyword analysis, and more.")
st.write("-----")


# load data
@st.cache_data
def load_data():
    df = pd.read_csv("raw_data/chatgpt_reviews - chatgpt_reviews.csv")
    return df

df = load_data()

nltk.download("vader_lexicon")
model = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    score = model.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["review"].apply(classify_sentiment)


option = st.selectbox("Select Analysis",[
    "Overall Sentiment Distribution",
    "Sentiment by Rating",
    "Keyword Analysis Word Cloud",
    "Sentiment Over Time",
    "Verified vs Non-Verified Users",
    "Review Length vs Sentiment",
    "Sentiment by Location",
    "Sentiment by Platform",
    "ChatGPT Version Sentiment",
    "Negative Feedback Themes"
])

# 1️⃣ Overall Sentiment Distribution
if option == "Overall Sentiment Distribution":
    st.title(" Overall Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%")
    st.pyplot(fig)

# 2️⃣ Sentiment by Rating
elif option == "Sentiment by Rating":
    st.title(" Sentiment by Rating")
    fig, ax = plt.subplots()
    sns.countplot(x=df["rating"], hue=df["sentiment"], ax=ax)
    st.pyplot(fig)

# 3️⃣ Keyword Analysis (Word Cloud)
elif option == "Keyword Analysis Word Cloud":
    st.title(" Word Cloud of Reviews")
    sentiment_choice = st.selectbox("Select Sentiment", ["Positive", "Neutral", "Negative"])
    text_data = " ".join(df[df["sentiment"] == sentiment_choice]["review"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# 4️⃣ Sentiment Over Time
elif option == "Sentiment Over Time":
    st.title(" Sentiment Trend Over Time")
    df["date"] = pd.to_datetime(df["date"])
    sentiment_trend = df.groupby(df["date"].dt.to_period("M"))["sentiment"].value_counts().unstack()
    fig, ax = plt.subplots()
    sentiment_trend.plot(kind="line", ax=ax)
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.title("Sentiment Over Time")
    st.pyplot(fig)

# 5️⃣ Verified vs Non-Verified Users
elif option == "Verified vs Non-Verified Users":
    st.title(" Verified vs Non-Verified Users Sentiment")
    fig, ax = plt.subplots()
    sns.barplot(x=df["verified_purchase"], y=df["sentiment"].apply(lambda x: 1 if x == "Positive" else -1 if x == "Negative" else 0), hue=df["verified_purchase"], ax=ax)
    plt.xlabel("Verified Purchase")
    plt.ylabel("Sentiment Score")
    st.pyplot(fig)

# 6️⃣ Review Length vs Sentiment
elif option == "Review Length vs Sentiment":
    st.title(" Review Length vs Sentiment")
    df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))
    fig, ax = plt.subplots()
    sns.boxplot(x=df["sentiment"], y=df["review_length"], ax=ax)
    plt.xlabel("Sentiment")
    plt.xticks(rotation=0)
    plt.ylabel("Review Length")
    st.pyplot(fig)

# 7️⃣ Sentiment by Location
elif option == "Sentiment by Location":
    st.title(" Sentiment by Location")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=df["location"].sort_values(ascending=False), y=df["sentiment"].apply(lambda x: 1 if x == "Positive" else -1 if x == "Negative" else 0),hue=df["location"], ax=ax)
    plt.xlabel("Location")
    plt.ylabel("Sentiment Score")
    st.pyplot(fig)

# 8️⃣ Sentiment by Platform
elif option == "Sentiment by Platform":
    st.title(" Sentiment by Platforms Web vs Mobile")
    fig, ax = plt.subplots()
    sns.barplot(x=df["platform"], y=df["sentiment"].apply(lambda x: 1 if x == "Positive" else -1 if x == "Negative" else 0), hue=df["platform"], ax=ax)
    plt.xlabel("Platform")
    plt.ylabel("Sentiment Score")
    st.pyplot(fig)

# 9️⃣ ChatGPT Version Sentiment
elif option == "ChatGPT Version Sentiment":
    st.title("ChatGPT Version Sentiment Analysis")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=df["version"], y=df["sentiment"].apply(lambda x: 1 if x == "Positive" else -1 if x == "Negative" else 0), hue=df["version"], ax=ax)
    plt.xlabel("ChatGPT Version")
    plt.ylabel("Sentiment Score")
    st.pyplot(fig)

# 🔟 Negative Feedback Themes
elif option == "Negative Feedback Themes":
    st.title(" Most Common Negative Feedback Themes")
    negative_text = " ".join(df[df["sentiment"] == "Negative"]["review"])
    wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(negative_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

st.write("-----")
st.warning("This dashboard is bit slow because of the large dataset. Please be patient while it loads.")