"""
Live Market Data Tool
Fetches real-time stock prices and fundamentals from Yahoo Finance (free, no API key).
"""

from langchain_core.tools import tool


# Map common Indian company names to Yahoo Finance tickers
TICKER_MAP = {
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "ril": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "tata consultancy services": "TCS.NS",
    "hdfc": "HDFCBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "wipro": "WIPRO.NS",
    "itc": "ITC.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS",
    "icici bank": "ICICIBANK.NS",
    "icici": "ICICIBANK.NS",
    "kotak": "KOTAKBANK.NS",
    "kotak mahindra": "KOTAKBANK.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "asian paints": "ASIANPAINT.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "hul": "HINDUNILVR.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "larsen": "LT.NS",
    "l&t": "LT.NS",
    "sun pharma": "SUNPHARMA.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "adani": "ADANIENT.NS",
    "adani enterprises": "ADANIENT.NS",
}


def resolve_ticker(company_name: str) -> str:
    """Resolve a company name to a Yahoo Finance ticker symbol."""
    name_lower = company_name.lower().strip()

    # Direct match
    if name_lower in TICKER_MAP:
        return TICKER_MAP[name_lower]

    # Partial match
    for key, ticker in TICKER_MAP.items():
        if key in name_lower or name_lower in key:
            return ticker

    # If it already looks like a ticker (e.g. "RELIANCE.NS"), use as-is
    if "." in company_name:
        return company_name

    # Last resort: try appending .NS (NSE)
    return f"{company_name.upper().replace(' ', '')}.NS"


@tool
def get_live_stock_data(company_name: str) -> str:
    """
    Fetch real-time stock price, valuation metrics, and financial fundamentals
    from Yahoo Finance. Completely free, no API key needed.

    Args:
        company_name: Name of the company (e.g. 'Reliance', 'TCS', 'Infosys')

    Returns:
        Formatted string with live market data including price, P/E, market cap,
        revenue, profit margins, and 52-week range.
    """
    try:
        import yfinance as yf

        ticker_symbol = resolve_ticker(company_name)
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            # Try without .NS suffix
            alt_ticker = company_name.upper().replace(" ", "")
            stock = yf.Ticker(alt_ticker)
            info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            return f"Could not fetch live data for '{company_name}'. Proceeding with document data."

        # Format currency helper
        def fmt_crore(val):
            if val is None:
                return "N/A"
            return f"₹{val / 10_000_000:,.0f} Cr"

        def fmt_price(val):
            if val is None:
                return "N/A"
            return f"₹{val:,.2f}"

        def fmt_pct(val):
            if val is None:
                return "N/A"
            return f"{val * 100:.1f}%"

        output = f"""=== LIVE MARKET DATA ({ticker_symbol}) ===
Company: {info.get('longName', company_name)}
Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}

PRICE & VALUATION
- Current Price: {fmt_price(info.get('currentPrice') or info.get('regularMarketPrice'))}
- Previous Close: {fmt_price(info.get('previousClose'))}
- Day Range: {fmt_price(info.get('dayLow'))} — {fmt_price(info.get('dayHigh'))}
- 52-Week Range: {fmt_price(info.get('fiftyTwoWeekLow'))} — {fmt_price(info.get('fiftyTwoWeekHigh'))}
- Market Cap: {fmt_crore(info.get('marketCap'))}

RATIOS & MULTIPLES
- P/E Ratio (Trailing): {info.get('trailingPE', 'N/A')}
- P/E Ratio (Forward): {info.get('forwardPE', 'N/A')}
- P/B Ratio: {info.get('priceToBook', 'N/A')}
- EV/EBITDA: {info.get('enterpriseToEbitda', 'N/A')}
- Dividend Yield: {fmt_pct(info.get('dividendYield'))}

FUNDAMENTALS
- Revenue (TTM): {fmt_crore(info.get('totalRevenue'))}
- Net Income (TTM): {fmt_crore(info.get('netIncomeToCommon'))}
- Profit Margin: {fmt_pct(info.get('profitMargins'))}
- Operating Margin: {fmt_pct(info.get('operatingMargins'))}
- Return on Equity: {fmt_pct(info.get('returnOnEquity'))}
- Debt-to-Equity: {info.get('debtToEquity', 'N/A')}
- Free Cash Flow: {fmt_crore(info.get('freeCashflow'))}

ANALYST CONSENSUS
- Target Mean Price: {fmt_price(info.get('targetMeanPrice'))}
- Target High: {fmt_price(info.get('targetHighPrice'))}
- Target Low: {fmt_price(info.get('targetLowPrice'))}
- Recommendation: {info.get('recommendationKey', 'N/A').upper()}
"""
        return output

    except Exception as e:
        return f"Live market data error for '{company_name}': {str(e)}. Proceeding with document data."
