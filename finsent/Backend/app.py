##coded by Noah Cowley, debugged by Claude

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import feedparser
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import quote
import os

app = Flask(__name__, static_folder='.')

CORS(app, origins=["*"])
CORS(app)



def fetch_news(query, num_articles=10):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)
    news_items = feed.entries[:num_articles]  # FIXED: was ["num_articles"]

    articles = []
    for item in news_items:
        title = item.title
        link = item.link
        published = item.published
        content = fetch_article_content(link)

        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "content": content
        })

    return articles  # FIXED: indentation was wrong


def fetch_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content.strip()
    except requests.RequestException:
        return "Content not retrieved."


def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    polarity = scores['compound']

    if polarity > 0.05:
        sentiment = 'Positive'
    elif polarity < -0.05:  # FIXED: spacing
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return polarity, sentiment


def analyze_ticker_sentiment(ticker, num_articles_per_query=10):
    queries = [
        f"{ticker} news",  # FIXED: missing comma
        f"{ticker} trends",
        f"{ticker} analysis",
        f"{ticker} forecast",
        f"{ticker} investment"
    ]

    all_articles = []

    for query in queries:
        articles = fetch_news(query, num_articles_per_query)
        all_articles.extend(articles)

    analyzed_articles = []
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for article in all_articles:
        polarity, sentiment = analyze_sentiment(article['title'])

        article_data = {
            'title': article['title'],
            'link': article['link'],
            'published': article['published'],
            'polarity': round(polarity, 3),
            'sentiment': sentiment
        }
        analyzed_articles.append(article_data)
        summary[sentiment] += 1

    total = len(analyzed_articles)
    summary_percent = {
        sentiment: {
            'count': count,
            'percentage': round((count / total) * 100, 2) if total > 0 else 0
        }
        for sentiment, count in summary.items()
    }

    return {
        'ticker': ticker,
        'articles': analyzed_articles,
        'summary': summary_percent,
        'total_articles': total
    }


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()

        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        num_articles = data.get('num_articles', 10)  # FIXED: was 'num_artiles'

        try:
            num_articles = int(num_articles)
            if num_articles < 1 or num_articles > 20:
                num_articles = 10
        except:
            num_articles = 10

        results = analyze_ticker_sentiment(ticker, num_articles)
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Sentiment Analysis API is working'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)