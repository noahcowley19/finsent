from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import feedparser
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import os
import time

app = Flask(__name__, static_folder='.')

CORS(app, origins=["*"])
CORS(app)


def fetch_article_content(url, timeout=5):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            script.decompose()
        
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs[:10] if p.get_text().strip()])
        
        if len(content) > 50:
            return content
        return ""
    except:
        return ""


def fetch_news(query, num_articles=10):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)
    news_items = feed.entries[:num_articles]

    articles = []
    
    def fetch_single_article(item):
        content = fetch_article_content(item.link)
        if not content:
            content = item.get('summary', '') or item.get('description', '')
        
        return {
            "title": item.title,
            "link": item.link,
            "published": item.published,
            "content": content
        }
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_item = {executor.submit(fetch_single_article, item): item for item in news_items}
        
        for future in as_completed(future_to_item, timeout=15):
            try:
                article = future.result(timeout=2)
                articles.append(article)
            except:
                item = future_to_item[future]
                articles.append({
                    "title": item.title,
                    "link": item.link,
                    "published": item.published,
                    "content": item.get('summary', '') or item.get('description', '')
                })
    
    return articles


def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    polarity = scores['compound']

    if polarity > 0.05:
        sentiment = 'Positive'
    elif polarity < -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return polarity, sentiment


def analyze_ticker_sentiment(ticker, num_articles_per_query=10):
    queries = [
        f"{ticker} stock news",
        f"{ticker} market analysis",
        f"{ticker} price forecast",
        f"{ticker} earnings",
        f"{ticker} investment outlook"
    ]

    all_articles = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_query = {
            executor.submit(fetch_news, query, num_articles_per_query): query 
            for query in queries
        }
        
        for future in as_completed(future_to_query, timeout=30):
            try:
                articles = future.result(timeout=5)
                all_articles.extend(articles)
            except:
                continue

    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title_lower = article['title'].lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_articles.append(article)

    analyzed_articles = []
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for article in unique_articles:
        text_to_analyze = f"{article['title']} {article['content']}"
        polarity, sentiment = analyze_sentiment(text_to_analyze)

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

        num_articles = data.get('num_articles', 10)

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
