from flask import Flask, render_template, request
import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import os

# Initialize Flask
app = Flask(__name__)

# Download VADER lexicon
nltk.download('vader_lexicon')

# ========== CONFIGURATION ==========
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAANNS5QEAAAAApC0KeOAM0qWtMEmHb9c9CCeGBZE%3DGJP3mApBrTSmW09HKx6xbcfHGaQEOksv5sUruvMvR5p5ZalpWu"

"

client = tweepy.Client(bearer_token="BEARER TOKEN")



# ========== ROUTES ==========

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    stock_name = request.form['stock']

    query = f"{stock_name} OR ${stock_name} lang:en -is:retweet"
    tweets = client.search_recent_tweets(query=query, max_results=100)

    if not tweets.data:
        return render_template('result.html', error="No tweets found. Try another stock name.")

    tweet_texts = [tweet.text for tweet in tweets.data]
    df = pd.DataFrame(tweet_texts, columns=['Tweet'])

    analyzer = SentimentIntensityAnalyzer()
    df['compound'] = df['Tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    def get_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['Sentiment'] = df['compound'].apply(get_sentiment)

    # Pie chart
    sentiment_counts = df['Sentiment'].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f"Sentiment Analysis for {stock_name}")
    plt.savefig("static/sentiment_pie.png")
    plt.close()

    # WordCloud
    all_words = ' '.join(df['Tweet'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Common Words for {stock_name}")
    plt.savefig("static/wordcloud.png")
    plt.close()

    avg_score = round(df['compound'].mean(), 3)

    return render_template('result.html', stock=stock_name, sentiment_counts=sentiment_counts.to_dict(), avg_score)
