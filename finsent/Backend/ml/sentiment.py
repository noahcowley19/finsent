# Sentiment Analyzer Module for Portfolio Holdings
# Uses VADER for news headline sentiment analysis

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf

# Import VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not installed. Sentiment analysis unavailable.")


def safe_round(value, decimals=2):
    """Safely round values, handling None and NaN"""
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    try:
        return round(value, decimals)
    except:
        return None


def get_ticker_news(ticker: str, max_articles: int = 10) -> List[Dict]:
    """
    Fetch news articles for a ticker using yfinance
    Returns list of news items with title, summary, and source
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return []
        
        articles = []
        for item in news[:max_articles]:
            articles.append({
                'title': item.get('title', ''),
                'summary': item.get('summary', ''),
                'publisher': item.get('publisher', ''),
                'link': item.get('link', ''),
                'published': datetime.fromtimestamp(
                    item.get('providerPublishTime', 0)
                ).isoformat() if item.get('providerPublishTime') else None,
            })
        
        return articles
        
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []


def analyze_text_sentiment(text: str) -> Dict:
    """
    Analyze sentiment of a text using VADER
    Returns compound score (-1 to +1) and classification
    """
    if not VADER_AVAILABLE or not text:
        return {'compound': 0, 'classification': 'neutral', 'confidence': 0}
    
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    compound = scores['compound']
    
    # Classify sentiment
    if compound >= 0.35:
        classification = 'very_positive'
    elif compound >= 0.15:
        classification = 'positive'
    elif compound >= -0.15:
        classification = 'neutral'
    elif compound >= -0.35:
        classification = 'negative'
    else:
        classification = 'very_negative'
    
    # Confidence based on how far from neutral
    confidence = abs(compound)
    
    return {
        'compound': safe_round(compound, 3),
        'positive': safe_round(scores['pos'], 3),
        'neutral': safe_round(scores['neu'], 3),
        'negative': safe_round(scores['neg'], 3),
        'classification': classification,
        'confidence': safe_round(confidence, 3),
    }


def analyze_ticker_sentiment(ticker: str) -> Dict:
    """
    Analyze overall sentiment for a ticker based on recent news
    Returns aggregated sentiment score and individual article sentiments
    """
    if not VADER_AVAILABLE:
        return {
            'ticker': ticker,
            'error': 'VADER not available',
            'sentiment_score': 0,
            'classification': 'neutral',
        }
    
    # Get news
    articles = get_ticker_news(ticker)
    
    if not articles:
        return {
            'ticker': ticker,
            'sentiment_score': 0,
            'classification': 'neutral',
            'confidence': 0,
            'articles_analyzed': 0,
            'note': 'No recent news found',
        }
    
    # Analyze each article
    article_sentiments = []
    compound_scores = []
    
    for article in articles:
        # Combine title and summary for analysis
        text = f"{article['title']} {article.get('summary', '')}"
        sentiment = analyze_text_sentiment(text)
        
        article_sentiments.append({
            'title': article['title'],
            'publisher': article['publisher'],
            'published': article['published'],
            'sentiment': sentiment,
        })
        
        compound_scores.append(sentiment['compound'])
    
    # Calculate aggregate sentiment
    if compound_scores:
        # Weight more recent articles higher
        weights = np.linspace(1, 1.5, len(compound_scores))[::-1]  # Most recent = highest weight
        weighted_score = np.average(compound_scores, weights=weights)
        
        # Classification of aggregate
        if weighted_score >= 0.25:
            classification = 'bullish'
        elif weighted_score >= 0.05:
            classification = 'slightly_bullish'
        elif weighted_score >= -0.05:
            classification = 'neutral'
        elif weighted_score >= -0.25:
            classification = 'slightly_bearish'
        else:
            classification = 'bearish'
        
        return {
            'ticker': ticker,
            'sentiment_score': safe_round(weighted_score, 3),
            'classification': classification,
            'confidence': safe_round(np.std(compound_scores), 3),  # Lower std = higher confidence
            'articles_analyzed': len(articles),
            'min_sentiment': safe_round(min(compound_scores), 3),
            'max_sentiment': safe_round(max(compound_scores), 3),
            'articles': article_sentiments[:5],  # Return top 5 articles
        }
    
    return {
        'ticker': ticker,
        'sentiment_score': 0,
        'classification': 'neutral',
        'confidence': 0,
        'articles_analyzed': 0,
    }


def analyze_portfolio_sentiment(positions: List[Dict]) -> Dict:
    """
    Analyze sentiment for all portfolio holdings
    Returns individual and weighted portfolio sentiment
    """
    if not VADER_AVAILABLE:
        return {
            'error': 'VADER not available',
            'portfolio_sentiment': 0,
        }
    
    holdings_sentiment = []
    weighted_scores = []
    total_value = sum(float(p.get('current_value', 0)) for p in positions)
    
    for pos in positions:
        ticker = pos.get('ticker', '').upper()
        current_value = float(pos.get('current_value', 0))
        weight = current_value / total_value if total_value > 0 else 0
        
        sentiment = analyze_ticker_sentiment(ticker)
        sentiment['weight'] = safe_round(weight * 100, 2)
        sentiment['current_value'] = current_value
        
        holdings_sentiment.append(sentiment)
        
        # Weight by portfolio allocation
        score = sentiment.get('sentiment_score', 0) or 0
        weighted_scores.append(score * weight)
    
    # Calculate portfolio-level sentiment
    portfolio_sentiment = sum(weighted_scores)
    
    # Classify portfolio sentiment
    if portfolio_sentiment >= 0.20:
        portfolio_classification = 'Very Bullish'
    elif portfolio_sentiment >= 0.08:
        portfolio_classification = 'Bullish'
    elif portfolio_sentiment >= -0.08:
        portfolio_classification = 'Neutral'
    elif portfolio_sentiment >= -0.20:
        portfolio_classification = 'Bearish'
    else:
        portfolio_classification = 'Very Bearish'
    
    # Sort holdings by sentiment score
    holdings_sentiment.sort(key=lambda x: x.get('sentiment_score', 0) or 0, reverse=True)
    
    return {
        'portfolio_sentiment': safe_round(portfolio_sentiment, 3),
        'portfolio_classification': portfolio_classification,
        'holdings_count': len(positions),
        'most_bullish': holdings_sentiment[0]['ticker'] if holdings_sentiment else None,
        'most_bearish': holdings_sentiment[-1]['ticker'] if holdings_sentiment else None,
        'holdings': holdings_sentiment,
        'timestamp': datetime.now().isoformat(),
    }


def get_sentiment_summary_emoji(classification: str) -> str:
    """Get emoji representation of sentiment"""
    mapping = {
        'Very Bullish': '🚀',
        'Bullish': '📈',
        'Neutral': '➡️',
        'Bearish': '📉',
        'Very Bearish': '⚠️',
        'bullish': '📈',
        'slightly_bullish': '↗️',
        'neutral': '➡️',
        'slightly_bearish': '↘️',
        'bearish': '📉',
    }
    return mapping.get(classification, '❓')
