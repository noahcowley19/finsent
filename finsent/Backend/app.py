import os
import re
import logging
from flask import Flask, request, abort, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Blueprints
from sentiment import sentiment_bp
from financials import financials_bp
from insider import insider_bp
from search import search_bp
from portfolio import portfolio_bp
from quant_lab import quant_lab_bp
from economic import economic_bp
from earnings import earnings_bp
from alerts import alerts_bp
from screener import screener_bp
from movers import movers_bp
from sectors import sectors_bp
from technicals import technicals_bp
from compare import compare_bp
from dividends import dividends_bp



app = Flask(__name__)

# Configure CORS origins from environment variable or use defaults
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    # Parse comma-separated origins from environment
    ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    # Default origins
    ALLOWED_ORIGINS = [
        "http://localhost:3000",  # Local development
        "https://caveray.com",     # Production domain
        "https://*.netlify.app",   # Netlify deployments
    ]

logger.info(f"CORS enabled for origins: {ALLOWED_ORIGINS}")

CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# Configure rate limiter
rate_limit = os.getenv("RATE_LIMIT", "15 per minute")
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[rate_limit] 
)


def get_client_ip():
    """
    Prioritize X-Forwarded-For (comma-separated IPs) and fall back to remote_addr.
    On platforms like Render / Netlify the real client IP is usually in X-Forwarded-For.
    """
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        # X-Forwarded-For: client, proxy1, proxy2
        # We want the left-most non-empty IP
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[0]
    # fallback
    return request.remote_addr or ""

DEFAULT_BAD_AGENTS = [
    "curl", "wget", "python-requests", "scrapy", "ahrefs", "mj12bot",
    "semrush", "bytespider", "gptbot", "perplexity", "facebookexternalhit",
    "python-urllib", "libwww-perl", "bingpreview", "pinterest", "baiduspider",
    "yandex", "sistrix", "dotbot", "slurp", "yahoo", "spbot"
]

bad_agents_env = os.getenv("BAD_AGENTS", "")
if bad_agents_env:
    BAD_AGENTS = [x.strip().lower() for x in bad_agents_env.split(",") if x.strip()]
else:
    BAD_AGENTS = DEFAULT_BAD_AGENTS

DEFAULT_BAD_AGENT_REGEX = [
    r"^curl\/",                # curl/X curl/7.XX
    r"^python-requests",       # python-requests/2.x
    r"^wget\/",               # wget/...
]

bad_agent_regex_env = os.getenv("BAD_AGENT_REGEX", "")
if bad_agent_regex_env:
    BAD_AGENT_REGEX = [r.strip() for r in bad_agent_regex_env.split(",") if r.strip()]
else:
    BAD_AGENT_REGEX = DEFAULT_BAD_AGENT_REGEX

BAD_AGENT_RE = [re.compile(p, re.IGNORECASE) for p in BAD_AGENT_REGEX]

@app.before_request
def block_bad_user_agents():
    ua = (request.headers.get("User-Agent") or "").lower()
    if not ua:
        # Many legitimate browsers send UA; empty UA often indicates a bot/script
        client_ip = get_client_ip()
        logger.info(f"Blocking request with empty User-Agent from {client_ip} path={request.path}")
        return _forbidden_json("Empty User-Agent blocked")

    for bad in BAD_AGENTS:
        if bad in ua:
            client_ip = get_client_ip()
            logger.info(f"Blocked UA substring match: '{bad}' UA='{ua[:120]}' ip={client_ip} path={request.path}")
            return _forbidden_json(f"Your user-agent is blocked: {bad}")

    for rex in BAD_AGENT_RE:
        if rex.search(ua):
            client_ip = get_client_ip()
            logger.info(f"Blocked UA regex match: '{rex.pattern}' UA='{ua[:120]}' ip={client_ip} path={request.path}")
            return _forbidden_json("Your user-agent is blocked")


DEFAULT_BLOCKED_IPS = [
    # exact IPs
    "123.45.67.89",
    # prefix-based (block beginswith)
    "45.146.164.",    # blocks any 45.146.164.* addresses
]

# You can override with BLOCKED_IPS environment variable (comma-separated)
blocked_ips_env = os.getenv("BLOCKED_IPS", "")
if blocked_ips_env:
    BLOCKED_IPS = [x.strip() for x in blocked_ips_env.split(",") if x.strip()]
else:
    BLOCKED_IPS = DEFAULT_BLOCKED_IPS

# Optional: CIDR-style blocking using ipaddress module for accuracy
CIDR_BLOCK_ENV = os.getenv("BLOCKED_CIDRS", "")  # comma separated CIDR ranges
CIDR_BLOCKS = []
if CIDR_BLOCK_ENV:
    import ipaddress
    for cidr in [c.strip() for c in CIDR_BLOCK_ENV.split(",") if c.strip()]:
        try:
            CIDR_BLOCKS.append(ipaddress.ip_network(cidr))
        except Exception:
            logger.warning(f"Invalid CIDR in BLOCKED_CIDRS: {cidr}")

@app.before_request
def block_bad_ips():
    ip = get_client_ip()
    if not ip:
        return  # can't identify client IP; let other protections handle it

    for blocked in BLOCKED_IPS:
        if blocked.endswith("."):
            # treat as prefix
            if ip.startswith(blocked):
                logger.info(f"Blocked IP prefix {blocked} matched client {ip} path={request.path}")
                return _forbidden_json("Access from your IP range is blocked")
        else:
            if ip == blocked:
                logger.info(f"Blocked exact IP {ip} path={request.path}")
                return _forbidden_json("Access from your IP is blocked")

    if CIDR_BLOCKS:
        import ipaddress
        try:
            ip_obj = ipaddress.ip_address(ip)
            for net in CIDR_BLOCKS:
                if ip_obj in net:
                    logger.info(f"Blocked CIDR {net} matched client {ip} path={request.path}")
                    return _forbidden_json("Access from your IP is blocked (CIDR)")
        except Exception:
            # invalid IP format — ignore
            pass

def _forbidden_json(message="Forbidden"):
    resp = jsonify({"error": "forbidden", "message": message})
    resp.status_code = 403
    # Optional: add a header for easier log filtering
    resp.headers["X-Blocked-By"] = "ua-ip-filter"
    return resp

@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if hasattr(e, 'code'):
        return jsonify({
            'error': str(e),
            'error_type': 'http_error'
        }), e.code

    if request.path.startswith('/api/'):
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500
    raise e


app.register_blueprint(sentiment_bp)
app.register_blueprint(financials_bp)
app.register_blueprint(insider_bp)
app.register_blueprint(search_bp)
app.register_blueprint(portfolio_bp)
app.register_blueprint(quant_lab_bp)
app.register_blueprint(economic_bp)
app.register_blueprint(earnings_bp)
app.register_blueprint(alerts_bp)
app.register_blueprint(screener_bp)
app.register_blueprint(movers_bp)
app.register_blueprint(sectors_bp)
app.register_blueprint(technicals_bp)
app.register_blueprint(compare_bp)
app.register_blueprint(dividends_bp)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)














