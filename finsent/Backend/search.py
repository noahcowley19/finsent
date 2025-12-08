
from flask import Blueprint, request, jsonify
import yfinance as yf
import feedparser
from datetime import datetime, timedelta
from urllib.parse import quote
import math
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

search_bp = Blueprint('search', __name__)

# Cache configuration
_search_cache = {}
_search_cache_time = {}
SEARCH_CACHE_TTL = 300  # 5 minutes

_market_movers_cache = {}
_market_movers_cache_time = 0
MARKET_MOVERS_CACHE_TTL = 180  # 3 minutes

# Sector ETF mappings for heatmap
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB',
    'Industrials': 'XLI',
    'Communication Services': 'XLC',
}

# Popular stocks for discovery
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'JPM', 'V', 'UNH', 'MA', 'HD', 'PG', 'JNJ', 'XOM', 'BAC', 'COST',
    'AVGO', 'MRK', 'ABBV', 'KO', 'PEP', 'CVX', 'WMT', 'LLY', 'ADBE',
    'CRM', 'TMO', 'CSCO', 'ACN', 'MCD', 'ABT', 'NKE', 'DHR', 'ORCL'
]


def clean_value(value):
    """Clean NaN/Inf values"""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def safe_round(value, decimals=2):
    """Safely round a value"""
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        return round(cleaned, decimals)
    except:
        return None


def format_large_number(value):
    """Format large numbers with T/B/M/K suffixes"""
    if value is None:
        return 'N/A'
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"${value / 1e3:.1f}K"
    else:
        return f"${value:,.0f}"


def format_number(value):
    """Format numbers without currency symbol"""
    if value is None:
        return 'N/A'
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{value / 1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"{value / 1e3:.1f}K"
    else:
        return f"{value:,.0f}"


def get_status(value, thresholds, inverse=False):
    """Determine status based on thresholds"""
    if value is None:
        return 'neutral'
    
    low, high = thresholds
    
    if inverse:
        if value >= high:
            return 'positive'
        elif value <= low:
            return 'negative'
        return 'neutral'
    else:
        if value <= low:
            return 'positive'
        elif value >= high:
            return 'negative'
        return 'neutral'


def get_stock_info(ticker):
    """Fetch comprehensive stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        quote_type = info.get('quoteType', '').upper()
        if quote_type not in ['EQUITY', 'STOCK', 'ETF', '']:
            return {
                'error': f'Search not available for {quote_type}. Please enter a stock or ETF ticker.',
                'error_type': 'invalid_asset'
            }
        
        if not info.get('shortName') and not info.get('longName'):
            return {
                'error': 'Ticker not recognized. Please enter a valid stock symbol.',
                'error_type': 'invalid_ticker'
            }
        
        return {
            'info': info,
            'stock': stock
        }
        
    except Exception as e:
        return {
            'error': f'Unable to fetch data for {ticker}. Please try again.',
            'error_type': 'fetch_error',
            'details': str(e)
        }


def get_company_overview(info):
    """Extract company overview data"""
    price = clean_value(info.get('currentPrice')) or clean_value(info.get('regularMarketPrice'))
    prev_close = clean_value(info.get('previousClose') or info.get('regularMarketPreviousClose'))
    
    change = None
    change_percent = None
    if price and prev_close:
        change = price - prev_close
        change_percent = (change / prev_close) * 100
    
    market_cap = clean_value(info.get('marketCap'))
    
    fifty_two_high = clean_value(info.get('fiftyTwoWeekHigh'))
    fifty_two_low = clean_value(info.get('fiftyTwoWeekLow'))
    
    range_position = None
    if price and fifty_two_high and fifty_two_low and fifty_two_high != fifty_two_low:
        range_position = ((price - fifty_two_low) / (fifty_two_high - fifty_two_low)) * 100
    
    # Calculate distance from 52-week high/low
    distance_from_high = None
    distance_from_low = None
    if price and fifty_two_high:
        distance_from_high = ((price - fifty_two_high) / fifty_two_high) * 100
    if price and fifty_two_low:
        distance_from_low = ((price - fifty_two_low) / fifty_two_low) * 100
    
    # Moving averages
    ma50 = clean_value(info.get('fiftyDayAverage'))
    ma200 = clean_value(info.get('twoHundredDayAverage'))
    
    # Price vs MAs
    above_ma50 = price > ma50 if price and ma50 else None
    above_ma200 = price > ma200 if price and ma200 else None
    
    return {
        'name': info.get('longName') or info.get('shortName') or 'Unknown',
        'ticker': info.get('symbol', '').upper(),
        'exchange': info.get('exchange', 'N/A'),
        'sector': info.get('sector') or 'N/A',
        'industry': info.get('industry') or 'N/A',
        'currency': info.get('currency', 'USD'),
        'price': price,
        'price_display': f"${price:.2f}" if price else 'N/A',
        'change': safe_round(change, 2),
        'change_percent': safe_round(change_percent, 2),
        'change_display': f"{'+' if change and change >= 0 else ''}{change:.2f}" if change else 'N/A',
        'change_percent_display': f"{'+' if change_percent and change_percent >= 0 else ''}{change_percent:.2f}%" if change_percent else 'N/A',
        'change_status': 'positive' if change and change >= 0 else 'negative' if change else 'neutral',
        'market_cap': market_cap,
        'market_cap_display': format_large_number(market_cap),
        'volume': clean_value(info.get('volume')),
        'volume_display': format_number(clean_value(info.get('volume'))),
        'avg_volume': clean_value(info.get('averageVolume')),
        'avg_volume_display': format_number(clean_value(info.get('averageVolume'))),
        'fifty_two_high': fifty_two_high,
        'fifty_two_high_display': f"${fifty_two_high:.2f}" if fifty_two_high else 'N/A',
        'fifty_two_low': fifty_two_low,
        'fifty_two_low_display': f"${fifty_two_low:.2f}" if fifty_two_low else 'N/A',
        'range_position': safe_round(range_position, 1),
        'distance_from_high': safe_round(distance_from_high, 2),
        'distance_from_low': safe_round(distance_from_low, 2),
        'day_high': clean_value(info.get('dayHigh')),
        'day_low': clean_value(info.get('dayLow')),
        'open': clean_value(info.get('open')),
        'prev_close': prev_close,
        'prev_close_display': f"${prev_close:.2f}" if prev_close else 'N/A',
        'beta': safe_round(clean_value(info.get('beta')), 2),
        'ma50': ma50,
        'ma50_display': f"${ma50:.2f}" if ma50 else 'N/A',
        'ma200': ma200,
        'ma200_display': f"${ma200:.2f}" if ma200 else 'N/A',
        'above_ma50': above_ma50,
        'above_ma200': above_ma200,
        'bid': clean_value(info.get('bid')),
        'ask': clean_value(info.get('ask')),
        'bid_size': clean_value(info.get('bidSize')),
        'ask_size': clean_value(info.get('askSize')),
    }


def get_valuation_metrics(info):
    """Extract valuation metrics"""
    metrics = []
    
    # P/E TTM
    pe_ttm = clean_value(info.get('trailingPE'))
    if pe_ttm and pe_ttm < 0:
        pe_ttm = None
    metrics.append({
        'name': 'P/E (TTM)',
        'value': pe_ttm,
        'display': f"{pe_ttm:.2f}" if pe_ttm else 'N/A',
        'status': get_status(pe_ttm, (15, 30)) if pe_ttm else 'neutral',
        'description': 'Price-to-Earnings ratio (trailing 12 months)'
    })
    
    # P/E Forward
    pe_fwd = clean_value(info.get('forwardPE'))
    if pe_fwd and pe_fwd < 0:
        pe_fwd = None
    metrics.append({
        'name': 'P/E (Fwd)',
        'value': pe_fwd,
        'display': f"{pe_fwd:.2f}" if pe_fwd else 'N/A',
        'status': get_status(pe_fwd, (12, 25)) if pe_fwd else 'neutral',
        'description': 'Forward Price-to-Earnings ratio'
    })
    
    # P/S
    ps = clean_value(info.get('priceToSalesTrailing12Months'))
    metrics.append({
        'name': 'P/S',
        'value': ps,
        'display': f"{ps:.2f}" if ps else 'N/A',
        'status': get_status(ps, (2, 8)) if ps else 'neutral',
        'description': 'Price-to-Sales ratio'
    })
    
    # P/B
    pb = clean_value(info.get('priceToBook'))
    if pb and pb < 0:
        pb = None
    metrics.append({
        'name': 'P/B',
        'value': pb,
        'display': f"{pb:.2f}" if pb else 'N/A',
        'status': get_status(pb, (1.5, 5)) if pb else 'neutral',
        'description': 'Price-to-Book ratio'
    })
    
    # EV/EBITDA
    ev_ebitda = clean_value(info.get('enterpriseToEbitda'))
    if ev_ebitda and ev_ebitda < 0:
        ev_ebitda = None
    metrics.append({
        'name': 'EV/EBITDA',
        'value': ev_ebitda,
        'display': f"{ev_ebitda:.2f}" if ev_ebitda else 'N/A',
        'status': get_status(ev_ebitda, (10, 20)) if ev_ebitda else 'neutral',
        'description': 'Enterprise Value to EBITDA'
    })
    
    # EV/Revenue
    ev_rev = clean_value(info.get('enterpriseToRevenue'))
    metrics.append({
        'name': 'EV/Revenue',
        'value': ev_rev,
        'display': f"{ev_rev:.2f}" if ev_rev else 'N/A',
        'status': get_status(ev_rev, (2, 8)) if ev_rev else 'neutral',
        'description': 'Enterprise Value to Revenue'
    })
    
    # PEG Ratio
    peg = clean_value(info.get('pegRatio'))
    metrics.append({
        'name': 'PEG',
        'value': peg,
        'display': f"{peg:.2f}" if peg else 'N/A',
        'status': get_status(peg, (1, 2)) if peg else 'neutral',
        'description': 'Price/Earnings to Growth ratio'
    })
    
    # Price to Free Cash Flow
    price = clean_value(info.get('currentPrice')) or clean_value(info.get('regularMarketPrice'))
    fcf_per_share = clean_value(info.get('freeCashflow'))
    shares = clean_value(info.get('sharesOutstanding'))
    p_fcf = None
    if price and fcf_per_share and shares and shares > 0:
        fcf_ps = fcf_per_share / shares
        if fcf_ps > 0:
            p_fcf = price / fcf_ps
    metrics.append({
        'name': 'P/FCF',
        'value': safe_round(p_fcf, 2),
        'display': f"{p_fcf:.2f}" if p_fcf else 'N/A',
        'status': get_status(p_fcf, (15, 30)) if p_fcf else 'neutral',
        'description': 'Price to Free Cash Flow'
    })
    
    return metrics


def get_profitability_metrics(info):
    """Extract profitability metrics"""
    metrics = []
    
    # Gross Margin
    gross_margin = clean_value(info.get('grossMargins'))
    if gross_margin:
        gross_margin = gross_margin * 100
    metrics.append({
        'name': 'Gross Margin',
        'value': gross_margin,
        'display': f"{gross_margin:.1f}%" if gross_margin else 'N/A',
        'status': get_status(gross_margin, (20, 40), inverse=True) if gross_margin else 'neutral',
        'description': 'Gross profit as % of revenue'
    })
    
    # Operating Margin
    op_margin = clean_value(info.get('operatingMargins'))
    if op_margin:
        op_margin = op_margin * 100
    metrics.append({
        'name': 'Operating Margin',
        'value': op_margin,
        'display': f"{op_margin:.1f}%" if op_margin else 'N/A',
        'status': get_status(op_margin, (10, 20), inverse=True) if op_margin else 'neutral',
        'description': 'Operating income as % of revenue'
    })
    
    # Net Margin
    net_margin = clean_value(info.get('profitMargins'))
    if net_margin:
        net_margin = net_margin * 100
    metrics.append({
        'name': 'Net Margin',
        'value': net_margin,
        'display': f"{net_margin:.1f}%" if net_margin else 'N/A',
        'status': get_status(net_margin, (5, 15), inverse=True) if net_margin else 'neutral',
        'description': 'Net income as % of revenue'
    })
    
    # EBITDA Margin
    ebitda = clean_value(info.get('ebitda'))
    revenue = clean_value(info.get('totalRevenue'))
    ebitda_margin = None
    if ebitda and revenue and revenue != 0:
        ebitda_margin = (ebitda / revenue) * 100
    metrics.append({
        'name': 'EBITDA Margin',
        'value': ebitda_margin,
        'display': f"{ebitda_margin:.1f}%" if ebitda_margin else 'N/A',
        'status': get_status(ebitda_margin, (15, 25), inverse=True) if ebitda_margin else 'neutral',
        'description': 'EBITDA as % of revenue'
    })
    
    # ROE
    roe = clean_value(info.get('returnOnEquity'))
    if roe:
        roe = roe * 100
    metrics.append({
        'name': 'ROE',
        'value': roe,
        'display': f"{roe:.1f}%" if roe else 'N/A',
        'status': get_status(roe, (10, 20), inverse=True) if roe else 'neutral',
        'description': 'Return on Equity'
    })
    
    # ROA
    roa = clean_value(info.get('returnOnAssets'))
    if roa:
        roa = roa * 100
    metrics.append({
        'name': 'ROA',
        'value': roa,
        'display': f"{roa:.1f}%" if roa else 'N/A',
        'status': get_status(roa, (5, 10), inverse=True) if roa else 'neutral',
        'description': 'Return on Assets'
    })
    
    # ROIC (calculated)
    metrics.append({
        'name': 'ROIC',
        'value': None,  # Would need balance sheet calculation
        'display': 'N/A',
        'status': 'neutral',
        'description': 'Return on Invested Capital'
    })
    
    return metrics


def get_financial_health(info):
    """Extract financial health metrics"""
    metrics = []
    
    # Current Ratio
    current_ratio = clean_value(info.get('currentRatio'))
    metrics.append({
        'name': 'Current Ratio',
        'value': current_ratio,
        'display': f"{current_ratio:.2f}" if current_ratio else 'N/A',
        'status': get_status(current_ratio, (1, 1.5), inverse=True) if current_ratio else 'neutral',
        'description': 'Current assets / current liabilities'
    })
    
    # Quick Ratio
    quick_ratio = clean_value(info.get('quickRatio'))
    metrics.append({
        'name': 'Quick Ratio',
        'value': quick_ratio,
        'display': f"{quick_ratio:.2f}" if quick_ratio else 'N/A',
        'status': get_status(quick_ratio, (0.8, 1.2), inverse=True) if quick_ratio else 'neutral',
        'description': '(Current assets - inventory) / current liabilities'
    })
    
    # Debt/Equity
    de = clean_value(info.get('debtToEquity'))
    if de:
        de = de / 100
    metrics.append({
        'name': 'Debt/Equity',
        'value': de,
        'display': f"{de:.2f}" if de else 'N/A',
        'status': get_status(de, (0.5, 1.5)) if de else 'neutral',
        'description': 'Total debt / shareholders equity'
    })
    
    # Total Debt
    total_debt = clean_value(info.get('totalDebt'))
    metrics.append({
        'name': 'Total Debt',
        'value': total_debt,
        'display': format_large_number(total_debt),
        'status': 'neutral',
        'description': 'Total debt outstanding'
    })
    
    # Total Cash
    total_cash = clean_value(info.get('totalCash'))
    metrics.append({
        'name': 'Total Cash',
        'value': total_cash,
        'display': format_large_number(total_cash),
        'status': 'neutral',
        'description': 'Cash and cash equivalents'
    })
    
    # Net Debt
    net_debt = None
    if total_debt is not None and total_cash is not None:
        net_debt = total_debt - total_cash
    metrics.append({
        'name': 'Net Debt',
        'value': net_debt,
        'display': format_large_number(net_debt) if net_debt else 'N/A',
        'status': 'positive' if net_debt and net_debt < 0 else 'negative' if net_debt and net_debt > 0 else 'neutral',
        'description': 'Total debt minus cash'
    })
    
    # Cash/Share
    cash_per_share = clean_value(info.get('totalCashPerShare'))
    metrics.append({
        'name': 'Cash/Share',
        'value': cash_per_share,
        'display': f"${cash_per_share:.2f}" if cash_per_share else 'N/A',
        'status': 'neutral',
        'description': 'Cash per share'
    })
    
    # Book Value/Share
    book_value = clean_value(info.get('bookValue'))
    metrics.append({
        'name': 'Book Value/Share',
        'value': book_value,
        'display': f"${book_value:.2f}" if book_value else 'N/A',
        'status': 'neutral',
        'description': 'Book value per share'
    })
    
    return metrics


def get_growth_metrics(info):
    """Extract growth metrics"""
    metrics = []
    
    # Revenue Growth YoY
    rev_growth = clean_value(info.get('revenueGrowth'))
    if rev_growth:
        rev_growth = rev_growth * 100
    metrics.append({
        'name': 'Revenue Growth (YoY)',
        'value': rev_growth,
        'display': f"{rev_growth:+.1f}%" if rev_growth else 'N/A',
        'status': 'positive' if rev_growth and rev_growth > 5 else 'negative' if rev_growth and rev_growth < 0 else 'neutral',
        'description': 'Year-over-year revenue growth'
    })
    
    # Earnings Growth YoY
    earn_growth = clean_value(info.get('earningsGrowth'))
    if earn_growth:
        earn_growth = earn_growth * 100
    metrics.append({
        'name': 'Earnings Growth (YoY)',
        'value': earn_growth,
        'display': f"{earn_growth:+.1f}%" if earn_growth else 'N/A',
        'status': 'positive' if earn_growth and earn_growth > 5 else 'negative' if earn_growth and earn_growth < 0 else 'neutral',
        'description': 'Year-over-year earnings growth'
    })
    
    # Quarterly Revenue Growth
    qtr_rev_growth = clean_value(info.get('revenueQuarterlyGrowth'))
    if qtr_rev_growth:
        qtr_rev_growth = qtr_rev_growth * 100
    metrics.append({
        'name': 'Revenue Growth (QoQ)',
        'value': qtr_rev_growth,
        'display': f"{qtr_rev_growth:+.1f}%" if qtr_rev_growth else 'N/A',
        'status': 'positive' if qtr_rev_growth and qtr_rev_growth > 0 else 'negative' if qtr_rev_growth and qtr_rev_growth < 0 else 'neutral',
        'description': 'Quarter-over-quarter revenue growth'
    })
    
    # Quarterly Earnings Growth
    qtr_earn_growth = clean_value(info.get('earningsQuarterlyGrowth'))
    if qtr_earn_growth:
        qtr_earn_growth = qtr_earn_growth * 100
    metrics.append({
        'name': 'Earnings Growth (QoQ)',
        'value': qtr_earn_growth,
        'display': f"{qtr_earn_growth:+.1f}%" if qtr_earn_growth else 'N/A',
        'status': 'positive' if qtr_earn_growth and qtr_earn_growth > 0 else 'negative' if qtr_earn_growth and qtr_earn_growth < 0 else 'neutral',
        'description': 'Quarter-over-quarter earnings growth'
    })
    
    # EPS TTM
    eps_ttm = clean_value(info.get('trailingEps'))
    metrics.append({
        'name': 'EPS (TTM)',
        'value': eps_ttm,
        'display': f"${eps_ttm:.2f}" if eps_ttm else 'N/A',
        'status': 'positive' if eps_ttm and eps_ttm > 0 else 'negative' if eps_ttm and eps_ttm < 0 else 'neutral',
        'description': 'Earnings per share (trailing 12 months)'
    })
    
    # EPS Forward
    eps_fwd = clean_value(info.get('forwardEps'))
    metrics.append({
        'name': 'EPS (Fwd)',
        'value': eps_fwd,
        'display': f"${eps_fwd:.2f}" if eps_fwd else 'N/A',
        'status': 'positive' if eps_fwd and eps_fwd > 0 else 'negative' if eps_fwd and eps_fwd < 0 else 'neutral',
        'description': 'Forward earnings per share estimate'
    })
    
    return metrics


def get_dividend_info(info):
    """Extract dividend information"""
    dividend_yield = clean_value(info.get('dividendYield'))
    if dividend_yield:
        dividend_yield = dividend_yield * 100
    
    dividend_rate = clean_value(info.get('dividendRate'))
    payout_ratio = clean_value(info.get('payoutRatio'))
    if payout_ratio:
        payout_ratio = payout_ratio * 100
    
    ex_date = info.get('exDividendDate')
    if ex_date:
        try:
            ex_date = datetime.fromtimestamp(ex_date).strftime('%b %d, %Y')
        except:
            ex_date = 'N/A'
    
    five_yr_avg = clean_value(info.get('fiveYearAvgDividendYield'))
    trailing_annual = clean_value(info.get('trailingAnnualDividendYield'))
    if trailing_annual:
        trailing_annual = trailing_annual * 100
    
    return {
        'has_dividend': dividend_yield is not None and dividend_yield > 0,
        'yield': dividend_yield,
        'yield_display': f"{dividend_yield:.2f}%" if dividend_yield else 'N/A',
        'rate': dividend_rate,
        'rate_display': f"${dividend_rate:.2f}" if dividend_rate else 'N/A',
        'payout_ratio': payout_ratio,
        'payout_ratio_display': f"{payout_ratio:.1f}%" if payout_ratio else 'N/A',
        'payout_status': get_status(payout_ratio, (30, 70)) if payout_ratio else 'neutral',
        'ex_date': ex_date or 'N/A',
        'five_yr_avg': five_yr_avg,
        'five_yr_avg_display': f"{five_yr_avg:.2f}%" if five_yr_avg else 'N/A',
        'trailing_annual': trailing_annual,
        'trailing_annual_display': f"{trailing_annual:.2f}%" if trailing_annual else 'N/A',
    }


def get_analyst_data(info):
    """Extract analyst ratings and price targets"""
    target_high = clean_value(info.get('targetHighPrice'))
    target_low = clean_value(info.get('targetLowPrice'))
    target_mean = clean_value(info.get('targetMeanPrice'))
    target_median = clean_value(info.get('targetMedianPrice'))
    
    current_price = clean_value(info.get('currentPrice')) or clean_value(info.get('regularMarketPrice'))
    
    upside = None
    upside_high = None
    upside_low = None
    
    if current_price and current_price > 0:
        if target_mean:
            upside = ((target_mean - current_price) / current_price) * 100
        if target_high:
            upside_high = ((target_high - current_price) / current_price) * 100
        if target_low:
            upside_low = ((target_low - current_price) / current_price) * 100
    
    recommendation = info.get('recommendationKey', 'N/A')
    recommendation_display = recommendation.replace('_', ' ').title() if recommendation != 'N/A' else 'N/A'
    
    num_analysts = clean_value(info.get('numberOfAnalystOpinions'))
    
    rec_status = 'neutral'
    if recommendation:
        rec_lower = recommendation.lower()
        if 'buy' in rec_lower or 'strong' in rec_lower:
            rec_status = 'positive'
        elif 'sell' in rec_lower or 'under' in rec_lower:
            rec_status = 'negative'
    
    # Recommendation mean (1=Strong Buy, 5=Strong Sell)
    rec_mean = clean_value(info.get('recommendationMean'))
    
    return {
        'has_data': target_mean is not None or recommendation != 'N/A',
        'target_high': target_high,
        'target_high_display': f"${target_high:.2f}" if target_high else 'N/A',
        'target_low': target_low,
        'target_low_display': f"${target_low:.2f}" if target_low else 'N/A',
        'target_mean': target_mean,
        'target_mean_display': f"${target_mean:.2f}" if target_mean else 'N/A',
        'target_median': target_median,
        'target_median_display': f"${target_median:.2f}" if target_median else 'N/A',
        'upside': safe_round(upside, 1),
        'upside_display': f"{upside:+.1f}%" if upside else 'N/A',
        'upside_high': safe_round(upside_high, 1),
        'upside_high_display': f"{upside_high:+.1f}%" if upside_high else 'N/A',
        'upside_low': safe_round(upside_low, 1),
        'upside_low_display': f"{upside_low:+.1f}%" if upside_low else 'N/A',
        'upside_status': 'positive' if upside and upside > 0 else 'negative' if upside and upside < 0 else 'neutral',
        'recommendation': recommendation,
        'recommendation_display': recommendation_display,
        'recommendation_status': rec_status,
        'recommendation_mean': rec_mean,
        'num_analysts': num_analysts,
        'num_analysts_display': str(int(num_analysts)) if num_analysts else 'N/A'
    }


def get_trading_info(info):
    """Extract trading information"""
    short_percent = clean_value(info.get('shortPercentOfFloat'))
    if short_percent:
        short_percent = short_percent * 100
    
    insider_percent = clean_value(info.get('heldPercentInsiders'))
    if insider_percent:
        insider_percent = insider_percent * 100
    
    institution_percent = clean_value(info.get('heldPercentInstitutions'))
    if institution_percent:
        institution_percent = institution_percent * 100
    
    return {
        'avg_volume_10d': clean_value(info.get('averageVolume10days')),
        'avg_volume_10d_display': format_number(clean_value(info.get('averageVolume10days'))),
        'avg_volume_3m': clean_value(info.get('averageVolume')),
        'avg_volume_3m_display': format_number(clean_value(info.get('averageVolume'))),
        'shares_outstanding': clean_value(info.get('sharesOutstanding')),
        'shares_outstanding_display': format_number(clean_value(info.get('sharesOutstanding'))),
        'float_shares': clean_value(info.get('floatShares')),
        'float_shares_display': format_number(clean_value(info.get('floatShares'))),
        'shares_short': clean_value(info.get('sharesShort')),
        'shares_short_display': format_number(clean_value(info.get('sharesShort'))),
        'short_ratio': safe_round(clean_value(info.get('shortRatio')), 2),
        'short_percent': safe_round(short_percent, 2),
        'short_percent_display': f"{short_percent:.2f}%" if short_percent else 'N/A',
        'insider_percent': safe_round(insider_percent, 2),
        'insider_percent_display': f"{insider_percent:.2f}%" if insider_percent else 'N/A',
        'institution_percent': safe_round(institution_percent, 2),
        'institution_percent_display': f"{institution_percent:.2f}%" if institution_percent else 'N/A',
    }


def get_company_profile(info):
    """Extract company profile"""
    employees = clean_value(info.get('fullTimeEmployees'))
    
    return {
        'description': info.get('longBusinessSummary', 'No description available.'),
        'website': info.get('website', 'N/A'),
        'employees': format_number(employees) if employees else 'N/A',
        'employees_raw': employees,
        'city': info.get('city', ''),
        'state': info.get('state', ''),
        'country': info.get('country', ''),
        'headquarters': ', '.join(filter(None, [
            info.get('city', ''),
            info.get('state', ''),
            info.get('country', '')
        ])) or 'N/A',
        'phone': info.get('phone', 'N/A'),
        'address': info.get('address1', ''),
        'zip': info.get('zip', ''),
    }


def get_key_stats_grid(info, overview):
    """Generate key stats grid for quick overview"""
    stats = []
    
    # Row 1: Core metrics
    stats.append({'label': 'Market Cap', 'value': overview['market_cap_display'], 'category': 'core'})
    stats.append({'label': 'P/E (TTM)', 'value': f"{clean_value(info.get('trailingPE')):.2f}" if clean_value(info.get('trailingPE')) else 'N/A', 'category': 'valuation'})
    stats.append({'label': 'EPS (TTM)', 'value': f"${clean_value(info.get('trailingEps')):.2f}" if clean_value(info.get('trailingEps')) else 'N/A', 'category': 'earnings'})
    stats.append({'label': 'Beta', 'value': f"{overview['beta']:.2f}" if overview['beta'] else 'N/A', 'category': 'risk'})
    
    # Row 2
    stats.append({'label': 'Revenue', 'value': format_large_number(clean_value(info.get('totalRevenue'))), 'category': 'fundamentals'})
    stats.append({'label': 'P/E (Fwd)', 'value': f"{clean_value(info.get('forwardPE')):.2f}" if clean_value(info.get('forwardPE')) else 'N/A', 'category': 'valuation'})
    stats.append({'label': 'EPS (Fwd)', 'value': f"${clean_value(info.get('forwardEps')):.2f}" if clean_value(info.get('forwardEps')) else 'N/A', 'category': 'earnings'})
    stats.append({'label': '52W Range', 'value': f"${overview['fifty_two_low']:.0f} - ${overview['fifty_two_high']:.0f}" if overview['fifty_two_low'] and overview['fifty_two_high'] else 'N/A', 'category': 'price'})
    
    # Row 3
    stats.append({'label': 'Net Income', 'value': format_large_number(clean_value(info.get('netIncomeToCommon'))), 'category': 'fundamentals'})
    stats.append({'label': 'P/S', 'value': f"{clean_value(info.get('priceToSalesTrailing12Months')):.2f}" if clean_value(info.get('priceToSalesTrailing12Months')) else 'N/A', 'category': 'valuation'})
    stats.append({'label': 'Book/Share', 'value': f"${clean_value(info.get('bookValue')):.2f}" if clean_value(info.get('bookValue')) else 'N/A', 'category': 'fundamentals'})
    stats.append({'label': 'Dividend', 'value': f"{clean_value(info.get('dividendYield'))*100:.2f}%" if clean_value(info.get('dividendYield')) else 'N/A', 'category': 'income'})
    
    # Row 4
    stats.append({'label': 'EBITDA', 'value': format_large_number(clean_value(info.get('ebitda'))), 'category': 'fundamentals'})
    stats.append({'label': 'P/B', 'value': f"{clean_value(info.get('priceToBook')):.2f}" if clean_value(info.get('priceToBook')) else 'N/A', 'category': 'valuation'})
    stats.append({'label': 'Cash/Share', 'value': f"${clean_value(info.get('totalCashPerShare')):.2f}" if clean_value(info.get('totalCashPerShare')) else 'N/A', 'category': 'fundamentals'})
    stats.append({'label': 'Payout', 'value': f"{clean_value(info.get('payoutRatio'))*100:.1f}%" if clean_value(info.get('payoutRatio')) else 'N/A', 'category': 'income'})
    
    # Row 5
    stats.append({'label': 'Volume', 'value': overview['volume_display'], 'category': 'trading'})
    stats.append({'label': 'EV/EBITDA', 'value': f"{clean_value(info.get('enterpriseToEbitda')):.2f}" if clean_value(info.get('enterpriseToEbitda')) else 'N/A', 'category': 'valuation'})
    stats.append({'label': 'Debt/Eq', 'value': f"{clean_value(info.get('debtToEquity'))/100:.2f}" if clean_value(info.get('debtToEquity')) else 'N/A', 'category': 'fundamentals'})
    stats.append({'label': 'ROE', 'value': f"{clean_value(info.get('returnOnEquity'))*100:.1f}%" if clean_value(info.get('returnOnEquity')) else 'N/A', 'category': 'profitability'})
    
    # Row 6
    stats.append({'label': 'Avg Volume', 'value': overview['avg_volume_display'], 'category': 'trading'})
    stats.append({'label': 'PEG', 'value': f"{clean_value(info.get('pegRatio')):.2f}" if clean_value(info.get('pegRatio')) else 'N/A', 'category': 'valuation'})
    stats.append({'label': 'Current Ratio', 'value': f"{clean_value(info.get('currentRatio')):.2f}" if clean_value(info.get('currentRatio')) else 'N/A', 'category': 'fundamentals'})
    stats.append({'label': 'ROA', 'value': f"{clean_value(info.get('returnOnAssets'))*100:.1f}%" if clean_value(info.get('returnOnAssets')) else 'N/A', 'category': 'profitability'})
    
    return stats


def get_historical_data(stock, period='1y', interval='1d'):
    """Fetch historical price data"""
    try:
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return None
        
        data = {
            'dates': [],
            'prices': [],
            'volumes': [],
            'highs': [],
            'lows': [],
            'opens': []
        }
        
        for date, row in hist.iterrows():
            if interval in ['1h', '30m', '15m', '5m', '1m']:
                date_str = date.strftime('%Y-%m-%d %H:%M')
            else:
                date_str = date.strftime('%Y-%m-%d')
            
            data['dates'].append(date_str)
            data['prices'].append(safe_round(clean_value(row.get('Close')), 2))
            data['volumes'].append(int(clean_value(row.get('Volume')) or 0))
            data['highs'].append(safe_round(clean_value(row.get('High')), 2))
            data['lows'].append(safe_round(clean_value(row.get('Low')), 2))
            data['opens'].append(safe_round(clean_value(row.get('Open')), 2))
        
        # Calculate performance metrics
        if len(data['prices']) >= 2 and data['prices'][0] and data['prices'][-1]:
            start_price = data['prices'][0]
            end_price = data['prices'][-1]
            period_change = end_price - start_price
            period_change_percent = (period_change / start_price) * 100
            data['period_change'] = safe_round(period_change, 2)
            data['period_change_percent'] = safe_round(period_change_percent, 2)
        else:
            data['period_change'] = None
            data['period_change_percent'] = None
        
        return data
        
    except Exception as e:
        print(f"Error getting historical data: {e}")
        return None


def get_news(ticker, num_articles=12):
    """Fetch news articles for a ticker"""
    try:
        queries = [
            f"{ticker} stock",
            f"{ticker} earnings",
            f"{ticker} news"
        ]
        
        all_articles = []
        seen_titles = set()
        
        for query in queries:
            try:
                rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(rss_url)
                
                for item in feed.entries[:6]:
                    title = item.get('title', '')
                    if title and title.lower() not in seen_titles:
                        seen_titles.add(title.lower())
                        
                        published = item.get('published', '')
                        try:
                            pub_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
                            pub_display = pub_date.strftime('%b %d, %Y')
                            pub_relative = get_relative_time(pub_date)
                        except:
                            pub_display = published[:16] if published else 'Unknown'
                            pub_relative = pub_display
                        
                        source = 'News'
                        if ' - ' in title:
                            parts = title.rsplit(' - ', 1)
                            if len(parts) == 2:
                                title = parts[0]
                                source = parts[1]
                        
                        all_articles.append({
                            'title': title,
                            'source': source,
                            'link': item.get('link', '#'),
                            'published': pub_display,
                            'published_relative': pub_relative
                        })
            except:
                continue
        
        return all_articles[:num_articles]
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def get_relative_time(dt):
    """Convert datetime to relative time string"""
    now = datetime.now()
    diff = now - dt
    
    if diff.days > 30:
        return dt.strftime('%b %d')
    elif diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours}h ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes}m ago"
    else:
        return "Just now"


def get_peers(info, ticker):
    """Get peer companies in the same sector/industry"""
    sector = info.get('sector', '')
    industry = info.get('industry', '')
    market_cap = clean_value(info.get('marketCap')) or 0
    
    # Define market cap tiers
    if market_cap >= 200e9:
        cap_tier = 'mega'
    elif market_cap >= 10e9:
        cap_tier = 'large'
    elif market_cap >= 2e9:
        cap_tier = 'mid'
    else:
        cap_tier = 'small'
    
    # Predefined peer groups by sector (simplified)
    sector_peers = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AVGO', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC', 'AMD'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'LLY', 'BMY'],
        'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'SCHW', 'AXP', 'V', 'MA'],
        'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG'],
        'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY'],
        'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS'],
        'Industrials': ['CAT', 'DE', 'UNP', 'RTX', 'BA', 'HON', 'UPS', 'GE', 'MMM'],
        'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'DD', 'NUE'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL'],
        'Real Estate': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'SPG', 'WELL'],
    }
    
    peers = sector_peers.get(sector, [])
    # Remove the current ticker from peers
    peers = [p for p in peers if p.upper() != ticker.upper()][:8]
    
    return {
        'sector': sector,
        'industry': industry,
        'market_cap_tier': cap_tier,
        'peer_tickers': peers
    }


def generate_signals(info, overview):
    """Generate trading signals and alerts"""
    signals = []
    
    price = overview.get('price')
    ma50 = overview.get('ma50')
    ma200 = overview.get('ma200')
    
    # MA signals
    if price and ma50 and ma200:
        if price > ma50 > ma200:
            signals.append({
                'type': 'trend',
                'status': 'positive',
                'title': 'Bullish Trend',
                'description': 'Price above 50-day and 200-day moving averages'
            })
        elif price < ma50 < ma200:
            signals.append({
                'type': 'trend',
                'status': 'negative',
                'title': 'Bearish Trend',
                'description': 'Price below 50-day and 200-day moving averages'
            })
        
        # Golden/Death Cross proximity
        if ma50 and ma200:
            ma_diff_pct = ((ma50 - ma200) / ma200) * 100
            if 0 < ma_diff_pct < 3:
                signals.append({
                    'type': 'crossover',
                    'status': 'positive',
                    'title': 'Golden Cross Forming',
                    'description': '50-day MA approaching above 200-day MA'
                })
            elif -3 < ma_diff_pct < 0:
                signals.append({
                    'type': 'crossover',
                    'status': 'negative',
                    'title': 'Death Cross Forming',
                    'description': '50-day MA approaching below 200-day MA'
                })
    
    # 52-week high/low proximity
    range_pos = overview.get('range_position')
    if range_pos is not None:
        if range_pos >= 95:
            signals.append({
                'type': 'price_level',
                'status': 'positive',
                'title': 'Near 52-Week High',
                'description': f'Trading at {range_pos:.1f}% of 52-week range'
            })
        elif range_pos <= 5:
            signals.append({
                'type': 'price_level',
                'status': 'negative',
                'title': 'Near 52-Week Low',
                'description': f'Trading at {range_pos:.1f}% of 52-week range'
            })
    
    # Volume signal
    volume = clean_value(info.get('volume'))
    avg_volume = clean_value(info.get('averageVolume'))
    if volume and avg_volume and avg_volume > 0:
        vol_ratio = volume / avg_volume
        if vol_ratio >= 2:
            signals.append({
                'type': 'volume',
                'status': 'neutral',
                'title': 'Unusual Volume',
                'description': f'Volume is {vol_ratio:.1f}x average'
            })
    
    # Valuation signals
    pe = clean_value(info.get('trailingPE'))
    if pe:
        if pe < 10:
            signals.append({
                'type': 'valuation',
                'status': 'positive',
                'title': 'Low P/E',
                'description': f'P/E ratio of {pe:.1f} may indicate undervaluation'
            })
        elif pe > 50:
            signals.append({
                'type': 'valuation',
                'status': 'warning',
                'title': 'High P/E',
                'description': f'P/E ratio of {pe:.1f} may indicate overvaluation'
            })
    
    # Short interest signal
    short_pct = clean_value(info.get('shortPercentOfFloat'))
    if short_pct:
        short_pct = short_pct * 100
        if short_pct >= 20:
            signals.append({
                'type': 'short_interest',
                'status': 'warning',
                'title': 'High Short Interest',
                'description': f'{short_pct:.1f}% of float is sold short'
            })
    
    # Analyst signal
    num_analysts = clean_value(info.get('numberOfAnalystOpinions'))
    rec_mean = clean_value(info.get('recommendationMean'))
    if num_analysts and num_analysts >= 5 and rec_mean:
        if rec_mean <= 2:
            signals.append({
                'type': 'analyst',
                'status': 'positive',
                'title': 'Strong Buy Consensus',
                'description': f'Analyst rating: {rec_mean:.1f}/5 ({int(num_analysts)} analysts)'
            })
        elif rec_mean >= 4:
            signals.append({
                'type': 'analyst',
                'status': 'negative',
                'title': 'Sell Consensus',
                'description': f'Analyst rating: {rec_mean:.1f}/5 ({int(num_analysts)} analysts)'
            })
    
    return signals[:6]


# ===== API ENDPOINTS =====

@search_bp.route('/api/search', methods=['POST'])
def search_stock():
    """Main search endpoint"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        
        if not ticker:
            return jsonify({
                'error': 'Ticker is required',
                'error_type': 'validation'
            }), 400
        
        # Check cache
        cache_key = ticker
        if cache_key in _search_cache:
            cache_age = (datetime.now() - _search_cache_time.get(cache_key, datetime.min)).total_seconds()
            if cache_age < SEARCH_CACHE_TTL:
                return jsonify(_search_cache[cache_key])
        
        stock_data = get_stock_info(ticker)
        
        if 'error' in stock_data:
            return jsonify(stock_data), 400
        
        info = stock_data['info']
        stock = stock_data['stock']
        
        overview = get_company_overview(info)
        peers = get_peers(info, ticker)
        signals = generate_signals(info, overview)
        
        response = {
            'ticker': ticker,
            'overview': overview,
            'key_stats': get_key_stats_grid(info, overview),
            'valuation': get_valuation_metrics(info),
            'profitability': get_profitability_metrics(info),
            'financial_health': get_financial_health(info),
            'growth': get_growth_metrics(info),
            'dividend': get_dividend_info(info),
            'analyst': get_analyst_data(info),
            'trading': get_trading_info(info),
            'profile': get_company_profile(info),
            'peers': peers,
            'signals': signals,
            'news': get_news(ticker),
            'timestamp': datetime.now().isoformat(),
            'data_freshness': 'real-time' if info.get('regularMarketTime') else 'delayed'
        }
        
        # Cache response
        _search_cache[cache_key] = response
        _search_cache_time[cache_key] = datetime.now()
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@search_bp.route('/api/search/chart', methods=['POST'])
def get_chart_data():
    """Get historical chart data"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        period = data.get('period', '1y')
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        # Map period to yfinance parameters
        period_map = {
            '1d': ('1d', '5m'),
            '5d': ('5d', '15m'),
            '1m': ('1mo', '1h'),
            '3m': ('3mo', '1d'),
            '6m': ('6mo', '1d'),
            'ytd': ('ytd', '1d'),
            '1y': ('1y', '1d'),
            '2y': ('2y', '1wk'),
            '5y': ('5y', '1wk'),
            'max': ('max', '1mo')
        }
        
        yf_period, interval = period_map.get(period, ('1y', '1d'))
        
        stock = yf.Ticker(ticker)
        hist_data = get_historical_data(stock, yf_period, interval)
        
        if not hist_data:
            return jsonify({'error': 'No historical data available'}), 404
        
        return jsonify({
            'ticker': ticker,
            'period': period,
            'data': hist_data
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error fetching chart data: {str(e)}'
        }), 500


@search_bp.route('/api/search/compare', methods=['POST'])
def compare_stocks():
    """Compare multiple stocks"""
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'At least 2 tickers required for comparison'}), 400
        
        if len(tickers) > 4:
            tickers = tickers[:4]  # Limit to 4 stocks
        
        results = []
        
        for ticker in tickers:
            ticker = ticker.strip().upper()
            stock_data = get_stock_info(ticker)
            
            if 'error' in stock_data:
                continue
            
            info = stock_data['info']
            overview = get_company_overview(info)
            
            results.append({
                'ticker': ticker,
                'name': overview['name'],
                'price': overview['price'],
                'price_display': overview['price_display'],
                'change_percent': overview['change_percent'],
                'change_status': overview['change_status'],
                'market_cap': overview['market_cap'],
                'market_cap_display': overview['market_cap_display'],
                'sector': overview['sector'],
                'pe': clean_value(info.get('trailingPE')),
                'pe_display': f"{clean_value(info.get('trailingPE')):.2f}" if clean_value(info.get('trailingPE')) else 'N/A',
                'ps': clean_value(info.get('priceToSalesTrailing12Months')),
                'pb': clean_value(info.get('priceToBook')),
                'roe': clean_value(info.get('returnOnEquity')),
                'roe_display': f"{clean_value(info.get('returnOnEquity'))*100:.1f}%" if clean_value(info.get('returnOnEquity')) else 'N/A',
                'revenue_growth': clean_value(info.get('revenueGrowth')),
                'gross_margin': clean_value(info.get('grossMargins')),
                'operating_margin': clean_value(info.get('operatingMargins')),
                'dividend_yield': clean_value(info.get('dividendYield')),
                'beta': clean_value(info.get('beta')),
                'fifty_two_high': overview['fifty_two_high'],
                'fifty_two_low': overview['fifty_two_low'],
                'range_position': overview['range_position'],
            })
        
        if len(results) < 2:
            return jsonify({'error': 'Could not fetch data for enough tickers'}), 400
        
        return jsonify({
            'tickers': [r['ticker'] for r in results],
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@search_bp.route('/api/search/compare/chart', methods=['POST'])
def compare_chart():
    """Get normalized chart data for comparison"""
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        period = data.get('period', '1y')
        
        if not tickers:
            return jsonify({'error': 'Tickers required'}), 400
        
        period_map = {
            '1m': '1mo',
            '3m': '3mo',
            '6m': '6mo',
            'ytd': 'ytd',
            '1y': '1y',
            '2y': '2y',
            '5y': '5y'
        }
        
        yf_period = period_map.get(period, '1y')
        
        results = {}
        dates = None
        
        for ticker in tickers[:4]:
            ticker = ticker.strip().upper()
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=yf_period, interval='1d')
                
                if hist.empty:
                    continue
                
                prices = hist['Close'].values
                # Normalize to 100
                if prices[0] and prices[0] > 0:
                    normalized = (prices / prices[0]) * 100
                    results[ticker] = normalized.tolist()
                    
                    if dates is None:
                        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
            except:
                continue
        
        return jsonify({
            'dates': dates or [],
            'series': results,
            'period': period
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error fetching comparison data: {str(e)}'
        }), 500


@search_bp.route('/api/search/movers', methods=['GET'])
def get_market_movers():
    """Get market movers (gainers, losers, most active)"""
    global _market_movers_cache, _market_movers_cache_time
    
    try:
        # Check cache
        if _market_movers_cache and (time.time() - _market_movers_cache_time) < MARKET_MOVERS_CACHE_TTL:
            return jsonify(_market_movers_cache)
        
        # Fetch data for popular tickers
        ticker_str = ' '.join(POPULAR_TICKERS)
        data = yf.download(ticker_str, period='2d', interval='1d', progress=False, threads=True)
        
        movers = []
        
        for ticker in POPULAR_TICKERS:
            try:
                if len(POPULAR_TICKERS) == 1:
                    close_data = data['Close']
                else:
                    close_data = data['Close'][ticker] if ticker in data['Close'].columns else None
                
                if close_data is None or close_data.empty or len(close_data) < 2:
                    continue
                
                current_price = clean_value(float(close_data.iloc[-1]))
                prev_price = clean_value(float(close_data.iloc[-2]))
                
                if not current_price or not prev_price:
                    continue
                
                change = current_price - prev_price
                pct_change = (change / prev_price) * 100
                
                # Get volume
                if len(POPULAR_TICKERS) == 1:
                    vol_data = data['Volume']
                else:
                    vol_data = data['Volume'][ticker] if ticker in data['Volume'].columns else None
                
                volume = clean_value(float(vol_data.iloc[-1])) if vol_data is not None and not vol_data.empty else 0
                
                movers.append({
                    'ticker': ticker,
                    'price': current_price,
                    'price_display': f"${current_price:.2f}",
                    'change': safe_round(change, 2),
                    'change_percent': safe_round(pct_change, 2),
                    'change_display': f"{'+' if pct_change >= 0 else ''}{pct_change:.2f}%",
                    'change_status': 'positive' if pct_change >= 0 else 'negative',
                    'volume': volume,
                    'volume_display': format_number(volume)
                })
            except Exception as e:
                continue
        
        # Sort into categories
        gainers = sorted([m for m in movers if m['change_percent'] > 0], key=lambda x: x['change_percent'], reverse=True)[:10]
        losers = sorted([m for m in movers if m['change_percent'] < 0], key=lambda x: x['change_percent'])[:10]
        most_active = sorted(movers, key=lambda x: x['volume'], reverse=True)[:10]
        
        result = {
            'gainers': gainers,
            'losers': losers,
            'most_active': most_active,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        _market_movers_cache = result
        _market_movers_cache_time = time.time()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Error fetching market movers: {str(e)}',
            'gainers': [],
            'losers': [],
            'most_active': []
        }), 500


@search_bp.route('/api/search/sector-heatmap', methods=['GET'])
def get_sector_heatmap():
    """Get sector performance heatmap data"""
    try:
        results = []
        
        for sector, etf in SECTOR_ETFS.items():
            try:
                stock = yf.Ticker(etf)
                hist = stock.history(period='5d', interval='1d')
                
                if hist.empty or len(hist) < 2:
                    continue
                
                current = clean_value(float(hist['Close'].iloc[-1]))
                prev = clean_value(float(hist['Close'].iloc[-2]))
                
                if current and prev:
                    change_pct = ((current - prev) / prev) * 100
                    results.append({
                        'sector': sector,
                        'etf': etf,
                        'change_percent': safe_round(change_pct, 2),
                        'status': 'positive' if change_pct >= 0 else 'negative'
                    })
            except:
                continue
        
        # Sort by performance
        results.sort(key=lambda x: x['change_percent'], reverse=True)
        
        return jsonify({
            'sectors': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error fetching sector data: {str(e)}',
            'sectors': []
        }), 500


@search_bp.route('/api/search/quick', methods=['POST'])
def quick_search():
    """Quick search for autocomplete - returns minimal data fast"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker required'}), 400
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info.get('shortName') and not info.get('longName'):
                return jsonify({'found': False})
            
            price = clean_value(info.get('currentPrice')) or clean_value(info.get('regularMarketPrice'))
            prev_close = clean_value(info.get('previousClose'))
            
            change_pct = None
            if price and prev_close:
                change_pct = ((price - prev_close) / prev_close) * 100
            
            return jsonify({
                'found': True,
                'ticker': ticker,
                'name': info.get('longName') or info.get('shortName'),
                'price': price,
                'price_display': f"${price:.2f}" if price else 'N/A',
                'change_percent': safe_round(change_pct, 2),
                'change_status': 'positive' if change_pct and change_pct >= 0 else 'negative' if change_pct else 'neutral',
                'sector': info.get('sector', 'N/A'),
                'market_cap': clean_value(info.get('marketCap')),
                'market_cap_display': format_large_number(clean_value(info.get('marketCap')))
            })
        except:
            return jsonify({'found': False})
        
    except Exception as e:
        return jsonify({'found': False, 'error': str(e)})


@search_bp.route('/api/search/health', methods=['GET'])
def search_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'stock_search',
        'cache_size': len(_search_cache),
        'timestamp': datetime.now().isoformat()
    })
