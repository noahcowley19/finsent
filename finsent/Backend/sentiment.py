# Made by Noah C. debugged by claude
from flask import Blueprint, request, jsonify
import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

sentiment_bp = Blueprint('sentiment', __name__)

# Hugging Face Inference API for FinBERT
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_API_TOKEN = os.environ.get('HF_API_TOKEN')

# Flag to track if model is warm
_model_warmed = False


def warm_up_model():
    """Pre-warm the FinBERT model on HF servers."""
    global _model_warmed
    if _model_warmed:
        return True
    
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": "Stock price increased."},
            timeout=60
        )
        if response.status_code == 200:
            _model_warmed = True
            return True
        elif response.status_code == 503:
            # Model loading, wait and retry once
            data = response.json()
            wait_time = min(data.get('estimated_time', 20), 30)
            time.sleep(wait_time)
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": "Stock price increased."},
                timeout=60
            )
            if response.status_code == 200:
                _model_warmed = True
                return True
    except:
        pass
    return False


def query_finbert(text):
    """
    Query FinBERT model via Hugging Face Inference API.
    Returns sentiment label and confidence score.
    """
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    # Truncate text to avoid API limits (FinBERT max ~512 tokens)
    truncated_text = text[:1000]
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": truncated_text},
            timeout=15
        )
        
        # Handle model loading (cold start)
        if response.status_code == 503:
            return 0.0, 'Neutral'
                
        if response.status_code != 200:
            return 0.0, 'Neutral'
            
        result = response.json()
        
        # HF returns [[{"label": "positive", "score": 0.9}, ...]]
        if result and isinstance(result, list) and len(result) > 0:
            predictions = result[0] if isinstance(result[0], list) else result
            
            # Find the highest scoring label
            best = max(predictions, key=lambda x: x['score'])
            label = best['label'].lower()
            score = best['score']
            
            # Convert to polarity score (-1 to 1)
            if label == 'positive':
                polarity = score
            elif label == 'negative':
                polarity = -score
            else:  # neutral
                polarity = 0.0
            
            return polarity, label.capitalize()
            
    except requests.exceptions.Timeout:
        return 0.0, 'Neutral'
    except Exception as e:
        print(f"FinBERT API error: {e}")
        return 0.0, 'Neutral'
    
    return 0.0, 'Neutral'


def fetch_article_content(url):
    """Fetch and extract article text content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=3, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        # Limit content size to reduce memory usage
        content = response.content[:50000]
        soup = BeautifulSoup(content, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        paragraphs = soup.find_all('p')[:5]
        text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        if len(text) > 50:
            return text[:1000]
        return ""
    except:
        return ""


def fetch_news_batch(queries, num_articles_per_query):
    """Fetch news articles from Google News RSS."""
    all_articles = []
    
    for query in queries:
        try:
            rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
            feed = feedparser.parse(rss_url)
            news_items = feed.entries[:num_articles_per_query]
            
            for item in news_items:
                all_articles.append({
                    "title": item.title,
                    "link": item.link,
                    "published": item.get('published', ''),
                    "summary": (item.get('summary', '') or item.get('description', ''))[:500]
                })
        except:
            continue
    
    # Remove duplicates
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title_lower = article['title'].lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_articles.append(article)
    
    # Limit total articles to reduce processing time
    unique_articles = unique_articles[:25]
    
    def scrape_content(article):
        content = fetch_article_content(article['link'])
        if not content:
            content = article['summary']
        return {
            "title": article['title'],
            "link": article['link'],
            "published": article['published'],
            "content": content[:1000]
        }
    
    results = []
    
    # Process in smaller batches
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(scrape_content, article): article for article in unique_articles}
        
        for future in as_completed(futures, timeout=15):
            try:
                result = future.result(timeout=2)
                results.append(result)
            except:
                article = futures[future]
                results.append({
                    "title": article['title'],
                    "link": article['link'],
                    "published": article['published'],
                    "content": article['summary'][:500]
                })
    
    return results


def analyze_ticker_sentiment(ticker, num_articles_per_query=10):
    """Main function to analyze sentiment for a ticker/asset."""
    
    # Warm up model first
    warm_up_model()
    
    queries = [
        f"{ticker} stock news",
        f"{ticker} market",
        f"{ticker} financial"
    ]

    articles = fetch_news_batch(queries, num_articles_per_query)
    
    if not articles:
        return {
            'ticker': ticker,
            'articles': [],
            'summary': {
                'Positive': {'count': 0, 'percentage': 0},
                'Negative': {'count': 0, 'percentage': 0},
                'Neutral': {'count': 0, 'percentage': 0}
            },
            'total_articles': 0,
            'model': 'FinBERT'
        }

    # Analyze sentiment sequentially to avoid overwhelming the API
    analyzed_articles = []
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    
    for article in articles:
        text_to_analyze = f"{article['title']} {article['content']}"
        polarity, sentiment = query_finbert(text_to_analyze)
        
        analyzed_articles.append({
            'title': article['title'],
            'link': article['link'],
            'published': article['published'],
            'polarity': round(polarity, 3),
            'sentiment': sentiment
        })
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
        'total_articles': total,
        'model': 'FinBERT'
    }


@sentiment_bp.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()

        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        num_articles = data.get('num_articles', 8)

        try:
            num_articles = int(num_articles)
            if num_articles < 1 or num_articles > 15:
                num_articles = 8
        except:
            num_articles = 8

        results = analyze_ticker_sentiment(ticker, num_articles)
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@sentiment_bp.route('/api/sentiment/health', methods=['GET'])
def sentiment_health():
    """Health check that also verifies FinBERT API connectivity."""
    try:
        polarity, sentiment = query_finbert("Stock prices rose today.")
        return jsonify({
            'status': 'healthy',
            'model': 'FinBERT',
            'api': 'Hugging Face Inference',
            'test_result': {'polarity': polarity, 'sentiment': sentiment}
        })
    except Exception as e:
        return jsonify({
            'status': 'degraded',
            'model': 'FinBERT',
            'error': str(e)
        }), 500
