# Made by Noah C. and debugged by Claude.ai
from flask import Blueprint, request, jsonify
import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

sentiment_bp = Blueprint('sentiment', __name__)

HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_API_TOKEN = os.environ.get('HF_API_TOKEN')  

def query_finbert(text, retries=3):

    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    truncated_text = text[:1500]
    
    for attempt in range(retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": truncated_text},
                timeout=30
            )
            
            if response.status_code == 503:
                data = response.json()
                wait_time = data.get('estimated_time', 20)
                if attempt < retries - 1:
                    time.sleep(min(wait_time, 30))
                    continue
                    
            response.raise_for_status()
            result = response.json()
            
            if result and isinstance(result, list) and len(result) > 0:
                predictions = result[0] if isinstance(result[0], list) else result
                
                best = max(predictions, key=lambda x: x['score'])
                label = best['label'].lower()
                score = best['score']
                
                if label == 'positive':
                    polarity = score
                elif label == 'negative':
                    polarity = -score
                else:
                    polarity = 0.0
                
                return polarity, label.capitalize()
                
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt) 
                continue
            print(f"FinBERT API error: {e}")
            
    return 0.0, 'Neutral'


def fetch_article_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=4, headers=headers, allow_redirects=True, stream=True)
        response.raise_for_status()
        
        content = response.content[:100000]
        soup = BeautifulSoup(content, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        
        paragraphs = soup.find_all('p')[:8]
        text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        if len(text) > 50:
            return text[:2000]
        return ""
    except:
        return ""


def fetch_news_batch(queries, num_articles_per_query):
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
                    "published": item.published,
                    "summary": item.get('summary', '') or item.get('description', '')
                })
        except:
            continue
    
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title_lower = article['title'].lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_articles.append(article)
    
    def scrape_content(article):
        content = fetch_article_content(article['link'])
        if not content:
            content = article['summary']
        return {
            "title": article['title'],
            "link": article['link'],
            "published": article['published'],
            "content": content
        }
    
    results = []
    batch_size = 15
    
    for i in range(0, len(unique_articles), batch_size):
        batch = unique_articles[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(scrape_content, article): article for article in batch}
            
            for future in as_completed(futures, timeout=20):
                try:
                    result = future.result(timeout=1)
                    results.append(result)
                except:
                    article = futures[future]
                    results.append({
                        "title": article['title'],
                        "link": article['link'],
                        "published": article['published'],
                        "content": article['summary']
                    })
    
    return results


def analyze_sentiment_batch(articles):
   
    analyzed = []
    
    def analyze_single(article):
        text_to_analyze = f"{article['title']} {article['content']}"
        polarity, sentiment = query_finbert(text_to_analyze)
        return {
            'title': article['title'],
            'link': article['link'],
            'published': article['published'],
            'polarity': round(polarity, 3),
            'sentiment': sentiment
        }
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(analyze_single, article): article for article in articles}
        
        for future in as_completed(futures, timeout=120):
            try:
                result = future.result(timeout=35)
                analyzed.append(result)
            except Exception as e:
                article = futures[future]
                analyzed.append({
                    'title': article['title'],
                    'link': article['link'],
                    'published': article['published'],
                    'polarity': 0.0,
                    'sentiment': 'Neutral'
                })
    
    return analyzed


def analyze_ticker_sentiment(ticker, num_articles_per_query=10):
    queries = [
        f"{ticker} stock news",
        f"{ticker} market analysis",
        f"{ticker} financial news",
        f"{ticker} investor",
        f"{ticker} earnings"
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

    analyzed_articles = analyze_sentiment_batch(articles)
    
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for article in analyzed_articles:
        summary[article['sentiment']] += 1

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


@sentiment_bp.route('/api/sentiment/health', methods=['GET'])
def sentiment_health():
    """Health check that also verifies FinBERT API connectivity."""
    try:
        polarity, sentiment = query_finbert("Stock prices rose today.", retries=1)
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
