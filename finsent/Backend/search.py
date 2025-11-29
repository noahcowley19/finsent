from flask import Blueprint, request, jsonify
import yfinance as yf
import feedparser
from datetime import datetime, timedelta
from urllib.parse import quote
import math
import pandas as pd

search_bp = Blueprint('search', __name__)

# In-memory cache for search data
_search_cache = {}
_search_cache_time = {}
SEARCH_CACHE_TTL = 300  # 5 minutes


def clean_value(value):
    """Clean NaN, Inf, and None values"""
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
    """Format large numbers with suffixes"""
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


def format_percent(value, multiply=False):
    """Format percentage values"""
    if value is None:
        return 'N/A'
    if multiply:
        value = value * 100
    return f"{value:.2f}%"


def get_status(value, thresholds, inverse=False):
    """
    Determine status based on thresholds.
    thresholds = (low, high) - values below low are positive, above high are negative
    inverse=True flips the logic
    """
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
    """Get comprehensive stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Validate ticker
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
    
    # 52-week data
    fifty_two_high = clean_value(info.get('fiftyTwoWeekHigh'))
    fifty_two_low = clean_value(info.get('fiftyTwoWeekLow'))
    
    # Calculate position in 52-week range
    range_position = None
    if price and fifty_two_high and fifty_two_low and fifty_two_high != fifty_two_low:
        range_position = ((price - fifty_two_low) / (fifty_two_high - fifty_two_low)) * 100
    
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
        'day_high': clean_value(info.get('dayHigh')),
        'day_low': clean_value(info.get('dayLow')),
        'open': clean_value(info.get('open')),
        'prev_close': prev_close,
        'beta': safe_round(clean_value(info.get('beta')), 2),
    }


def get_valuation_metrics(info):
    """Extract valuation metrics"""
    metrics = []
    
    # P/E Ratio (TTM)
    pe_ttm = clean_value(info.get('trailingPE'))
    if pe_ttm and pe_ttm < 0:
        pe_ttm = None
    metrics.append({
        'name': 'P/E (TTM)',
        'value': pe_ttm,
        'display': f"{pe_ttm:.2f}" if pe_ttm else 'N/A',
        'status': get_status(pe_ttm, (15, 30)) if pe_ttm else 'neutral'
    })
    
    # P/E Ratio (Forward)
    pe_fwd = clean_value(info.get('forwardPE'))
    if pe_fwd and pe_fwd < 0:
        pe_fwd = None
    metrics.append({
        'name': 'P/E (Fwd)',
        'value': pe_fwd,
        'display': f"{pe_fwd:.2f}" if pe_fwd else 'N/A',
        'status': get_status(pe_fwd, (12, 25)) if pe_fwd else 'neutral'
    })
    
    # P/S Ratio
    ps = clean_value(info.get('priceToSalesTrailing12Months'))
    metrics.append({
        'name': 'P/S',
        'value': ps,
        'display': f"{ps:.2f}" if ps else 'N/A',
        'status': get_status(ps, (2, 8)) if ps else 'neutral'
    })
    
    # P/B Ratio
    pb = clean_value(info.get('priceToBook'))
    if pb and pb < 0:
        pb = None
    metrics.append({
        'name': 'P/B',
        'value': pb,
        'display': f"{pb:.2f}" if pb else 'N/A',
        'status': get_status(pb, (1.5, 5)) if pb else 'neutral'
    })
    
    # EV/EBITDA
    ev_ebitda = clean_value(info.get('enterpriseToEbitda'))
    if ev_ebitda and ev_ebitda < 0:
        ev_ebitda = None
    metrics.append({
        'name': 'EV/EBITDA',
        'value': ev_ebitda,
        'display': f"{ev_ebitda:.2f}" if ev_ebitda else 'N/A',
        'status': get_status(ev_ebitda, (10, 20)) if ev_ebitda else 'neutral'
    })
    
    # EV/Revenue
    ev_rev = clean_value(info.get('enterpriseToRevenue'))
    metrics.append({
        'name': 'EV/Revenue',
        'value': ev_rev,
        'display': f"{ev_rev:.2f}" if ev_rev else 'N/A',
        'status': get_status(ev_rev, (2, 8)) if ev_rev else 'neutral'
    })
    
    # PEG Ratio
    peg = clean_value(info.get('pegRatio'))
    metrics.append({
        'name': 'PEG',
        'value': peg,
        'display': f"{peg:.2f}" if peg else 'N/A',
        'status': get_status(peg, (1, 2)) if peg else 'neutral'
    })
    
    # Trailing PEG
    trailing_peg = clean_value(info.get('trailingPegRatio'))
    metrics.append({
        'name': 'PEG (TTM)',
        'value': trailing_peg,
        'display': f"{trailing_peg:.2f}" if trailing_peg else 'N/A',
        'status': get_status(trailing_peg, (1, 2)) if trailing_peg else 'neutral'
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
        'status': get_status(gross_margin, (20, 40), inverse=True) if gross_margin else 'neutral'
    })
    
    # Operating Margin
    op_margin = clean_value(info.get('operatingMargins'))
    if op_margin:
        op_margin = op_margin * 100
    metrics.append({
        'name': 'Operating Margin',
        'value': op_margin,
        'display': f"{op_margin:.1f}%" if op_margin else 'N/A',
        'status': get_status(op_margin, (10, 20), inverse=True) if op_margin else 'neutral'
    })
    
    # Net Margin
    net_margin = clean_value(info.get('profitMargins'))
    if net_margin:
        net_margin = net_margin * 100
    metrics.append({
        'name': 'Net Margin',
        'value': net_margin,
        'display': f"{net_margin:.1f}%" if net_margin else 'N/A',
        'status': get_status(net_margin, (5, 15), inverse=True) if net_margin else 'neutral'
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
        'status': get_status(ebitda_margin, (15, 25), inverse=True) if ebitda_margin else 'neutral'
    })
    
    # ROE
    roe = clean_value(info.get('returnOnEquity'))
    if roe:
        roe = roe * 100
    metrics.append({
        'name': 'ROE',
        'value': roe,
        'display': f"{roe:.1f}%" if roe else 'N/A',
        'status': get_status(roe, (10, 20), inverse=True) if roe else 'neutral'
    })
    
    # ROA
    roa = clean_value(info.get('returnOnAssets'))
    if roa:
        roa = roa * 100
    metrics.append({
        'name': 'ROA',
        'value': roa,
        'display': f"{roa:.1f}%" if roa else 'N/A',
        'status': get_status(roa, (5, 10), inverse=True) if roa else 'neutral'
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
        'status': get_status(current_ratio, (1, 1.5), inverse=True) if current_ratio else 'neutral'
    })
    
    # Quick Ratio
    quick_ratio = clean_value(info.get('quickRatio'))
    metrics.append({
        'name': 'Quick Ratio',
        'value': quick_ratio,
        'display': f"{quick_ratio:.2f}" if quick_ratio else 'N/A',
        'status': get_status(quick_ratio, (0.8, 1.2), inverse=True) if quick_ratio else 'neutral'
    })
    
    # Debt to Equity
    de = clean_value(info.get('debtToEquity'))
    if de:
        de = de / 100  # yfinance returns as percentage
    metrics.append({
        'name': 'Debt/Equity',
        'value': de,
        'display': f"{de:.2f}" if de else 'N/A',
        'status': get_status(de, (0.5, 1.5)) if de else 'neutral'
    })
    
    # Total Debt
    total_debt = clean_value(info.get('totalDebt'))
    metrics.append({
        'name': 'Total Debt',
        'value': total_debt,
        'display': format_large_number(total_debt),
        'status': 'neutral'
    })
    
    # Total Cash
    total_cash = clean_value(info.get('totalCash'))
    metrics.append({
        'name': 'Total Cash',
        'value': total_cash,
        'display': format_large_number(total_cash),
        'status': 'neutral'
    })
    
    # Cash per Share
    cash_per_share = clean_value(info.get('totalCashPerShare'))
    metrics.append({
        'name': 'Cash/Share',
        'value': cash_per_share,
        'display': f"${cash_per_share:.2f}" if cash_per_share else 'N/A',
        'status': 'neutral'
    })
    
    return metrics


def get_growth_metrics(info):
    """Extract growth metrics"""
    metrics = []
    
    # Revenue Growth
    rev_growth = clean_value(info.get('revenueGrowth'))
    if rev_growth:
        rev_growth = rev_growth * 100
    metrics.append({
        'name': 'Revenue Growth (YoY)',
        'value': rev_growth,
        'display': f"{rev_growth:+.1f}%" if rev_growth else 'N/A',
        'status': 'positive' if rev_growth and rev_growth > 5 else 'negative' if rev_growth and rev_growth < 0 else 'neutral'
    })
    
    # Earnings Growth
    earn_growth = clean_value(info.get('earningsGrowth'))
    if earn_growth:
        earn_growth = earn_growth * 100
    metrics.append({
        'name': 'Earnings Growth (YoY)',
        'value': earn_growth,
        'display': f"{earn_growth:+.1f}%" if earn_growth else 'N/A',
        'status': 'positive' if earn_growth and earn_growth > 5 else 'negative' if earn_growth and earn_growth < 0 else 'neutral'
    })
    
    # Quarterly Revenue Growth
    qtr_rev_growth = clean_value(info.get('revenueQuarterlyGrowth'))
    if qtr_rev_growth:
        qtr_rev_growth = qtr_rev_growth * 100
    metrics.append({
        'name': 'Revenue Growth (QoQ)',
        'value': qtr_rev_growth,
        'display': f"{qtr_rev_growth:+.1f}%" if qtr_rev_growth else 'N/A',
        'status': 'positive' if qtr_rev_growth and qtr_rev_growth > 0 else 'negative' if qtr_rev_growth and qtr_rev_growth < 0 else 'neutral'
    })
    
    # Quarterly Earnings Growth
    qtr_earn_growth = clean_value(info.get('earningsQuarterlyGrowth'))
    if qtr_earn_growth:
        qtr_earn_growth = qtr_earn_growth * 100
    metrics.append({
        'name': 'Earnings Growth (QoQ)',
        'value': qtr_earn_growth,
        'display': f"{qtr_earn_growth:+.1f}%" if qtr_earn_growth else 'N/A',
        'status': 'positive' if qtr_earn_growth and qtr_earn_growth > 0 else 'negative' if qtr_earn_growth and qtr_earn_growth < 0 else 'neutral'
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
        'five_yr_avg_display': f"{five_yr_avg:.2f}%" if five_yr_avg else 'N/A'
    }


def get_analyst_data(info):
    """Extract analyst ratings and price targets"""
    target_high = clean_value(info.get('targetHighPrice'))
    target_low = clean_value(info.get('targetLowPrice'))
    target_mean = clean_value(info.get('targetMeanPrice'))
    target_median = clean_value(info.get('targetMedianPrice'))
    
    current_price = clean_value(info.get('currentPrice')) or clean_value(info.get('regularMarketPrice'))
    
    upside = None
    if target_mean and current_price and current_price > 0:
        upside = ((target_mean - current_price) / current_price) * 100
    
    recommendation = info.get('recommendationKey', 'N/A')
    recommendation_display = recommendation.replace('_', ' ').title() if recommendation != 'N/A' else 'N/A'
    
    num_analysts = clean_value(info.get('numberOfAnalystOpinions'))
    
    # Recommendation status
    rec_status = 'neutral'
    if recommendation:
        rec_lower = recommendation.lower()
        if 'buy' in rec_lower or 'strong' in rec_lower:
            rec_status = 'positive'
        elif 'sell' in rec_lower or 'under' in rec_lower:
            rec_status = 'negative'
    
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
        'upside_status': 'positive' if upside and upside > 0 else 'negative' if upside and upside < 0 else 'neutral',
        'recommendation': recommendation,
        'recommendation_display': recommendation_display,
        'recommendation_status': rec_status,
        'num_analysts': num_analysts,
        'num_analysts_display': str(int(num_analysts)) if num_analysts else 'N/A'
    }


def get_trading_info(info):
    """Extract trading information"""
    return {
        'avg_volume_10d': format_number(clean_value(info.get('averageVolume10days'))),
        'avg_volume_3m': format_number(clean_value(info.get('averageVolume'))),
        'shares_outstanding': format_number(clean_value(info.get('sharesOutstanding'))),
        'float_shares': format_number(clean_value(info.get('floatShares'))),
        'shares_short': format_number(clean_value(info.get('sharesShort'))),
        'short_ratio': safe_round(clean_value(info.get('shortRatio')), 2),
        'short_percent': safe_round(clean_value(info.get('shortPercentOfFloat')) * 100, 2) if clean_value(info.get('shortPercentOfFloat')) else None,
        'insider_percent': safe_round(clean_value(info.get('heldPercentInsiders')) * 100, 2) if clean_value(info.get('heldPercentInsiders')) else None,
        'institution_percent': safe_round(clean_value(info.get('heldPercentInstitutions')) * 100, 2) if clean_value(info.get('heldPercentInstitutions')) else None,
    }


def get_company_profile(info):
    """Extract company profile information"""
    employees = clean_value(info.get('fullTimeEmployees'))
    
    return {
        'description': info.get('longBusinessSummary', 'No description available.'),
        'website': info.get('website', 'N/A'),
        'employees': format_number(employees) if employees else 'N/A',
        'city': info.get('city', ''),
        'state': info.get('state', ''),
        'country': info.get('country', ''),
        'headquarters': ', '.join(filter(None, [
            info.get('city', ''),
            info.get('state', ''),
            info.get('country', '')
        ])) or 'N/A'
    }


def get_key_stats_grid(info, overview):
    """Create a Finviz-style key statistics grid"""
    stats = []
    
    # Row 1
    stats.append({'label': 'Market Cap', 'value': overview['market_cap_display']})
    stats.append({'label': 'P/E (TTM)', 'value': f"{clean_value(info.get('trailingPE')):.2f}" if clean_value(info.get('trailingPE')) else 'N/A'})
    stats.append({'label': 'EPS (TTM)', 'value': f"${clean_value(info.get('trailingEps')):.2f}" if clean_value(info.get('trailingEps')) else 'N/A'})
    stats.append({'label': 'Beta', 'value': f"{overview['beta']:.2f}" if overview['beta'] else 'N/A'})
    
    # Row 2
    stats.append({'label': 'Revenue', 'value': format_large_number(clean_value(info.get('totalRevenue')))})
    stats.append({'label': 'P/E (Fwd)', 'value': f"{clean_value(info.get('forwardPE')):.2f}" if clean_value(info.get('forwardPE')) else 'N/A'})
    stats.append({'label': 'EPS (Fwd)', 'value': f"${clean_value(info.get('forwardEps')):.2f}" if clean_value(info.get('forwardEps')) else 'N/A'})
    stats.append({'label': '52W Range', 'value': f"${overview['fifty_two_low']:.0f} - ${overview['fifty_two_high']:.0f}" if overview['fifty_two_low'] and overview['fifty_two_high'] else 'N/A'})
    
    # Row 3
    stats.append({'label': 'Net Income', 'value': format_large_number(clean_value(info.get('netIncomeToCommon')))})
    stats.append({'label': 'P/S', 'value': f"{clean_value(info.get('priceToSalesTrailing12Months')):.2f}" if clean_value(info.get('priceToSalesTrailing12Months')) else 'N/A'})
    stats.append({'label': 'Book/Share', 'value': f"${clean_value(info.get('bookValue')):.2f}" if clean_value(info.get('bookValue')) else 'N/A'})
    stats.append({'label': 'Dividend', 'value': f"{clean_value(info.get('dividendYield'))*100:.2f}%" if clean_value(info.get('dividendYield')) else 'N/A'})
    
    # Row 4
    stats.append({'label': 'EBITDA', 'value': format_large_number(clean_value(info.get('ebitda')))})
    stats.append({'label': 'P/B', 'value': f"{clean_value(info.get('priceToBook')):.2f}" if clean_value(info.get('priceToBook')) else 'N/A'})
    stats.append({'label': 'Cash/Share', 'value': f"${clean_value(info.get('totalCashPerShare')):.2f}" if clean_value(info.get('totalCashPerShare')) else 'N/A'})
    stats.append({'label': 'Payout', 'value': f"{clean_value(info.get('payoutRatio'))*100:.1f}%" if clean_value(info.get('payoutRatio')) else 'N/A'})
    
    # Row 5
    stats.append({'label': 'Volume', 'value': overview['volume_display']})
    stats.append({'label': 'EV/EBITDA', 'value': f"{clean_value(info.get('enterpriseToEbitda')):.2f}" if clean_value(info.get('enterpriseToEbitda')) else 'N/A'})
    stats.append({'label': 'Debt/Eq', 'value': f"{clean_value(info.get('debtToEquity'))/100:.2f}" if clean_value(info.get('debtToEquity')) else 'N/A'})
    stats.append({'label': 'ROE', 'value': f"{clean_value(info.get('returnOnEquity'))*100:.1f}%" if clean_value(info.get('returnOnEquity')) else 'N/A'})
    
    # Row 6
    stats.append({'label': 'Avg Volume', 'value': overview['avg_volume_display']})
    stats.append({'label': 'PEG', 'value': f"{clean_value(info.get('pegRatio')):.2f}" if clean_value(info.get('pegRatio')) else 'N/A'})
    stats.append({'label': 'Current Ratio', 'value': f"{clean_value(info.get('currentRatio')):.2f}" if clean_value(info.get('currentRatio')) else 'N/A'})
    stats.append({'label': 'ROA', 'value': f"{clean_value(info.get('returnOnAssets'))*100:.1f}%" if clean_value(info.get('returnOnAssets')) else 'N/A'})
    
    return stats


def get_historical_data(stock, period='1y', interval='1d'):
    """Get historical price data for charting"""
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
            # Format date based on interval
            if interval in ['1h', '30m', '15m', '5m']:
                date_str = date.strftime('%Y-%m-%d %H:%M')
            else:
                date_str = date.strftime('%Y-%m-%d')
            
            data['dates'].append(date_str)
            data['prices'].append(safe_round(clean_value(row.get('Close')), 2))
            data['volumes'].append(int(clean_value(row.get('Volume')) or 0))
            data['highs'].append(safe_round(clean_value(row.get('High')), 2))
            data['lows'].append(safe_round(clean_value(row.get('Low')), 2))
            data['opens'].append(safe_round(clean_value(row.get('Open')), 2))
        
        return data
        
    except Exception as e:
        print(f"Error getting historical data: {e}")
        return None


def get_news(ticker, num_articles=10):
    """Fetch news from Google News RSS"""
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
                
                for item in feed.entries[:5]:
                    title = item.get('title', '')
                    if title and title.lower() not in seen_titles:
                        seen_titles.add(title.lower())
                        
                        # Parse published date
                        published = item.get('published', '')
                        try:
                            pub_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
                            pub_display = pub_date.strftime('%b %d, %Y')
                            pub_relative = get_relative_time(pub_date)
                        except:
                            pub_display = published[:16] if published else 'Unknown'
                            pub_relative = pub_display
                        
                        # Extract source from title (usually "Title - Source")
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
        
        # Get stock data
        stock_data = get_stock_info(ticker)
        
        if 'error' in stock_data:
            return jsonify(stock_data), 400
        
        info = stock_data['info']
        stock = stock_data['stock']
        
        # Build response
        overview = get_company_overview(info)
        
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
            'news': get_news(ticker),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the response
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
    """Get historical data for charting"""
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


@search_bp.route('/api/search/health', methods=['GET'])
def search_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'stock_search'
    })
