"""
SWOT Analysis Service
======================
Fetches real-time data from multiple sources (yfinance, RSS news feeds,
financial ratios, peer comparisons) and uses Groq LLM to synthesize a
professional, data-driven SWOT analysis.

Data pipeline:
  1. yfinance  → fundamentals, price action, analyst recs, earnings, news
  2. RSS feeds → latest news headlines from Yahoo Finance / Google News
  3. Peer data → comparison vs sector competitors
  4. Groq LLM  → structured SWOT synthesis from all gathered data
"""

from __future__ import annotations

import os
import json
import math
import datetime
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import yfinance as yf
import numpy as np
import pandas as pd
import time
import random

# Groq client (reuse pattern from ai_services.py)
try:
    from groq import Groq
    _groq_key = os.environ.get("GROQ_API_KEY")
    _groq_client = Groq(api_key=_groq_key) if _groq_key else None
except Exception:
    _groq_client = None


# ──────────────────────────────────────────────────────────────────────
# Market helpers
# ──────────────────────────────────────────────────────────────────────

INDIA_SUFFIXES = {".NS", ".BO"}
INDIA_NAMES = {
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS",
    "infy": "INFY.NS", "hdfc": "HDFCBANK.NS", "icici": "ICICIBANK.NS",
    "wipro": "WIPRO.NS", "airtel": "BHARTIARTL.NS", "itc": "ITC.NS",
    "sbi": "SBIN.NS", "maruti": "MARUTI.NS", "tatamotors": "TATAMOTORS.NS",
    "bajaj": "BAJFINANCE.NS", "hindalco": "HINDALCO.NS", "titan": "TITAN.NS",
    "adani": "ADANIENT.NS", "sunpharma": "SUNPHARMA.NS", "drreddy": "DRREDDY.NS",
}

US_NAMES = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "tesla": "TSLA",
    "nvidia": "NVDA", "meta": "META", "netflix": "NFLX",
    "jpmorgan": "JPM", "berkshire": "BRK-B", "visa": "V",
    "walmart": "WMT", "johnson": "JNJ", "exxon": "XOM",
}


def resolve_ticker(query: str) -> tuple[str, str]:
    """
    Resolve a company name or ticker to a yfinance ticker symbol.
    Returns (ticker_symbol, market) where market is 'IN' or 'US'.
    """
    q = query.strip()
    q_lower = q.lower().replace(" ", "").replace(".", "")

    # Direct suffix match
    q_upper = q.upper().strip()
    for suf in INDIA_SUFFIXES:
        if q_upper.endswith(suf):
            return q_upper, "IN"

    # Name lookup - India
    for key, ticker in INDIA_NAMES.items():
        if key in q_lower:
            return ticker, "IN"

    # Name lookup - US
    for key, ticker in US_NAMES.items():
        if key in q_lower:
            return ticker, "US"

    # If it looks like a plain ticker (all caps, no spaces)
    if q_upper == q_upper.strip() and " " not in q_upper and len(q_upper) <= 6:
        market = "IN" if ".NS" in q_upper or ".BO" in q_upper else "US"
        return q_upper, market

    # Fallback: treat as US ticker
    return q_upper, "US"


# ──────────────────────────────────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────────────────────────────────

def _safe_float(val, default=None):
    try:
        return float(val) if val is not None and val != "N/A" else default
    except (TypeError, ValueError):
        return default


def _fmt_large(val, currency_sym="$"):
    """Format large numbers to readable strings (e.g. 1.2T, 450B, 35M)."""
    if val is None:
        return "N/A"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if val >= 1e12:
        return f"{currency_sym}{val/1e12:.2f}T"
    elif val >= 1e9:
        return f"{currency_sym}{val/1e9:.2f}B"
    elif val >= 1e6:
        return f"{currency_sym}{val/1e6:.2f}M"
    return f"{currency_sym}{val:,.0f}"


def fetch_fundamentals(ticker: str, market: str) -> Dict:
    """Fetch company fundamentals, price metrics, and financial ratios."""
    currency = "₹" if market == "IN" else "$"
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        name = (info.get("longName") or info.get("shortName") or ticker)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        country = info.get("country", "India" if market == "IN" else "United States")
        website = info.get("website", "N/A")
        description = (info.get("longBusinessSummary") or "")[:600]

        price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        prev_close = _safe_float(info.get("previousClose"))
        day_change = ((price - prev_close) / prev_close * 100) if price and prev_close else None

        market_cap = _safe_float(info.get("marketCap"))
        pe_ratio = _safe_float(info.get("trailingPE"))
        pb_ratio = _safe_float(info.get("priceToBook"))
        eps = _safe_float(info.get("trailingEps"))
        roe = _safe_float(info.get("returnOnEquity"))
        roa = _safe_float(info.get("returnOnAssets"))
        revenue = _safe_float(info.get("totalRevenue"))
        net_income = _safe_float(info.get("netIncomeToCommon"))
        profit_margin = _safe_float(info.get("profitMargins"))
        revenue_growth = _safe_float(info.get("revenueGrowth"))
        earnings_growth = _safe_float(info.get("earningsGrowth"))
        debt_equity = _safe_float(info.get("debtToEquity"))
        current_ratio = _safe_float(info.get("currentRatio"))
        total_debt = _safe_float(info.get("totalDebt"))
        free_cashflow = _safe_float(info.get("freeCashflow"))
        dividend_yield = _safe_float(info.get("dividendYield"))
        beta = _safe_float(info.get("beta"))
        week52_high = _safe_float(info.get("fiftyTwoWeekHigh"))
        week52_low = _safe_float(info.get("fiftyTwoWeekLow"))
        analyst_target = _safe_float(info.get("targetMeanPrice"))
        recommendation = info.get("recommendationKey", "N/A")
        employees = info.get("fullTimeEmployees", "N/A")
        exchange = info.get("exchange", "N/A")

        # Analyst ratings
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                recent_recs = recs.tail(5)
                rec_summary = json.loads(recent_recs.to_json(orient="records", date_format="iso"))
            else:
                rec_summary = []
        except Exception:
            rec_summary = []

        # Insider transactions (recent)
        try:
            insider = t.insider_transactions
            if insider is not None and not insider.empty:
                insider_summary_df = insider.head(3)
                insider_summary = json.loads(insider_summary_df.to_json(orient="records", date_format="iso"))
            else:
                insider_summary = []
        except Exception:
            insider_summary = []

        return {
            "name": name,
            "ticker": ticker,
            "market": market,
            "currency": currency,
            "sector": sector,
            "industry": industry,
            "country": country,
            "website": website,
            "description": description,
            "exchange": exchange,
            "employees": employees,
            "price": price,
            "prev_close": prev_close,
            "day_change_pct": day_change,
            "market_cap": market_cap,
            "market_cap_fmt": _fmt_large(market_cap, currency),
            "pe_ratio": pe_ratio,
            "pb_ratio": pb_ratio,
            "eps": eps,
            "roe": roe,
            "roa": roa,
            "revenue": revenue,
            "revenue_fmt": _fmt_large(revenue, currency),
            "net_income": net_income,
            "net_income_fmt": _fmt_large(net_income, currency),
            "profit_margin": profit_margin,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "debt_equity": debt_equity,
            "current_ratio": current_ratio,
            "total_debt": total_debt,
            "total_debt_fmt": _fmt_large(total_debt, currency),
            "free_cashflow": free_cashflow,
            "free_cashflow_fmt": _fmt_large(free_cashflow, currency),
            "dividend_yield": dividend_yield,
            "beta": beta,
            "week52_high": week52_high,
            "week52_low": week52_low,
            "analyst_target": analyst_target,
            "recommendation": recommendation,
            "analyst_recs": rec_summary,
            "insider_transactions": insider_summary,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker, "market": market, "currency": currency}


def fetch_price_metrics(ticker: str) -> Dict:
    """Fetch price history, compute technicals and vol metrics."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        if hist.empty:
            return {}

        close = hist["Close"]
        returns = close.pct_change().dropna()

        # 52-week performance
        perf_52w = float((close.iloc[-1] / close.iloc[0] - 1) * 100) if len(close) > 1 else 0.0

        # 30-day, 90-day performance
        perf_30d = float((close.iloc[-1] / close.iloc[-min(22, len(close))] - 1) * 100)
        perf_90d = float((close.iloc[-1] / close.iloc[-min(66, len(close))] - 1) * 100)

        # Historical vol
        ann_vol = float(returns.std() * math.sqrt(252) * 100)

        # Average volume
        avg_vol = float(hist["Volume"].tail(30).mean()) if "Volume" in hist.columns else None

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        # SMA
        sma_50 = float(close.rolling(50).mean().iloc[-1])
        sma_200_series = close.rolling(200).mean()
        sma_200 = float(sma_200_series.iloc[-1]) if not pd.isna(sma_200_series.iloc[-1]) else None

        current = float(close.iloc[-1])
        above_sma50 = current > sma_50
        above_sma200 = (current > sma_200) if sma_200 else None

        return {
            "perf_52w": round(perf_52w, 2),
            "perf_30d": round(perf_30d, 2),
            "perf_90d": round(perf_90d, 2),
            "ann_volatility_pct": round(ann_vol, 2),
            "rsi_14": round(rsi, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2) if sma_200 else None,
            "above_sma50": above_sma50,
            "above_sma200": above_sma200,
            "avg_volume_30d": avg_vol,
            "close_prices": close.tail(60).tolist(),
            "dates": [str(d.date()) for d in close.tail(60).index],
        }
    except Exception:
        return {}


def fetch_news(ticker: str, company_name: str) -> List[Dict]:
    """
    Fetch recent news from:
    1. yfinance news feed
    2. Yahoo Finance RSS feed
    Returns up to 15 news items with title, source, date, and summary.
    """
    news_items = []

    # Source 1: yfinance news
    try:
        t = yf.Ticker(ticker)
        raw = t.news or []
        for n in raw[:8]:
            content = n.get("content", n)
            title = content.get("title") or n.get("title", "")
            publisher = (content.get("provider", {}).get("displayName")
                         or n.get("publisher", "Unknown"))
            link = (content.get("canonicalUrl", {}).get("url") or n.get("link", "#"))
            pub_time = content.get("pubDate") or n.get("providerPublishTime", "")
            summary = content.get("summary") or ""
            if title:
                news_items.append({
                    "title": title,
                    "source": publisher,
                    "link": link,
                    "date": str(pub_time)[:10] if pub_time else "Recent",
                    "summary": summary[:200],
                })
    except Exception:
        pass

    # Source 2: Yahoo Finance RSS
    try:
        query = company_name.replace(" ", "+")
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(rss_url, headers=headers, timeout=6)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.iter("item"):
                title = (item.find("title").text or "").strip()
                link = (item.find("link").text or "#").strip()
                pub_date = (item.find("pubDate").text or "")[:16]
                desc = (item.find("description").text or "")[:200]
                if title and title not in [n["title"] for n in news_items]:
                    news_items.append({
                        "title": title, "source": "Yahoo Finance RSS",
                        "link": link, "date": pub_date, "summary": desc,
                    })
                if len(news_items) >= 15:
                    break
    except Exception:
        pass

    return news_items[:15]


def fetch_peers(ticker: str, sector: str, market: str) -> List[Dict]:
    """Fetch 3–5 sector peers and compare key financial ratios."""
    peer_map = {
        # Indian sectors
        "Technology": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
        "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS"],
        "Consumer Cyclical": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "HEROMOTOCO.NS"],
        "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "LUPIN.NS"],
        "Consumer Defensive": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
        # US sectors
        "Communication Services": ["GOOGL", "META", "NFLX", "SNAP", "PINS"],
        "Consumer Cyclical_US": ["AMZN", "TSLA", "NKE", "HD", "MCD"],
        "Technology_US": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"],
        "Financials_US": ["JPM", "BAC", "WFC", "GS", "MS"],
    }

    # Pick peer list
    if market == "US":
        key = sector + "_US" if (sector + "_US") in peer_map else sector
    else:
        key = sector
    candidates = peer_map.get(key, [])
    peers = [p for p in candidates if p.upper() != ticker.upper()][:4]

    peer_data = []
    for p in peers:
        try:
            info = yf.Ticker(p).info or {}
            peer_data.append({
                "ticker": p,
                "name": info.get("shortName", p),
                "market_cap": _fmt_large(info.get("marketCap"), "₹" if market == "IN" else "$"),
                "pe": round(info.get("trailingPE", 0) or 0, 1),
                "roe": round((info.get("returnOnEquity") or 0) * 100, 1),
                "revenue_growth": round((info.get("revenueGrowth") or 0) * 100, 1),
                "profit_margin": round((info.get("profitMargins") or 0) * 100, 1),
            })
        except Exception:
            continue
    return peer_data


# ──────────────────────────────────────────────────────────────────────
# SWOT Generation via Groq LLM
# ──────────────────────────────────────────────────────────────────────

def _build_prompt(fundamentals: Dict, price_metrics: Dict,
                  news: List[Dict], peers: List[Dict]) -> str:
    """Build a rich, data-packed prompt for the LLM."""
    curr = fundamentals.get("currency", "$")
    name = fundamentals.get("name", fundamentals.get("ticker", "Company"))
    ticker = fundamentals.get("ticker", "")
    market = fundamentals.get("market", "US")
    mkt_label = "India (NSE/BSE)" if market == "IN" else "US (NYSE/NASDAQ)"

    # News block (cleaned)
    news_block = "\n".join(
        f"[{n['date']}] {n['title']} — {n['source']}" +
        (f"\n   Summary: {n['summary']}" if n.get("summary") else "")
        for n in news[:12]
    ) or "No recent news available."

    # Peer comparison block
    peer_block = ""
    if peers:
        peer_block = "COMPETITOR BENCHMARKING:\n"
        for p in peers:
            peer_block += (
                f"  • {p['name']} ({p['ticker']}): MarketCap={p['market_cap']}, "
                f"P/E={p['pe']}, ROE={p['roe']}%, RevGrowth={p['revenue_growth']}%, "
                f"ProfitMargin={p['profit_margin']}%\n"
            )

    # Price metrics block
    pm = price_metrics
    price_block = f"""
PRICE & MARKET PERFORMANCE:
  • 52W Performance: {pm.get('perf_52w', 'N/A')}%
  • 30D Performance: {pm.get('perf_30d', 'N/A')}%
  • 90D Performance: {pm.get('perf_90d', 'N/A')}%
  • Annual Volatility: {pm.get('ann_volatility_pct', 'N/A')}%
  • RSI (14): {pm.get('rsi_14', 'N/A')}
  • Above SMA50: {pm.get('above_sma50', 'N/A')}
  • Above SMA200: {pm.get('above_sma200', 'N/A')}
  • Avg Volume (30D): {pm.get('avg_volume_30d', 'N/A'):,.0f}
""" if pm else ""

    # Fundamentals block
    fin_block = f"""
COMPANY: {name} ({ticker}) | Market: {mkt_label} | Currency: {curr}
Sector: {fundamentals.get('sector','N/A')} | Industry: {fundamentals.get('industry','N/A')}
Country: {fundamentals.get('country','N/A')} | Employees: {fundamentals.get('employees','N/A')}
Description: {fundamentals.get('description','')}

FINANCIAL METRICS:
  • Current Price: {curr}{fundamentals.get('price','N/A')}
  • Market Cap: {fundamentals.get('market_cap_fmt','N/A')}
  • P/E Ratio (TTM): {fundamentals.get('pe_ratio','N/A')}
  • P/B Ratio: {fundamentals.get('pb_ratio','N/A')}
  • EPS (TTM): {curr}{fundamentals.get('eps','N/A')}
  • ROE: {f"{fundamentals['roe']*100:.1f}%" if fundamentals.get('roe') else 'N/A'}
  • ROA: {f"{fundamentals['roa']*100:.1f}%" if fundamentals.get('roa') else 'N/A'}
  • Revenue: {fundamentals.get('revenue_fmt','N/A')}
  • Net Income: {fundamentals.get('net_income_fmt','N/A')}
  • Profit Margin: {f"{fundamentals['profit_margin']*100:.1f}%" if fundamentals.get('profit_margin') else 'N/A'}
  • Revenue Growth (YoY): {f"{fundamentals['revenue_growth']*100:.1f}%" if fundamentals.get('revenue_growth') else 'N/A'}
  • Earnings Growth (YoY): {f"{fundamentals['earnings_growth']*100:.1f}%" if fundamentals.get('earnings_growth') else 'N/A'}
  • Debt/Equity: {fundamentals.get('debt_equity','N/A')}
  • Current Ratio: {fundamentals.get('current_ratio','N/A')}
  • Total Debt: {fundamentals.get('total_debt_fmt','N/A')}
  • Free Cash Flow: {fundamentals.get('free_cashflow_fmt','N/A')}
  • Dividend Yield: {f"{fundamentals['dividend_yield']*100:.2f}%" if fundamentals.get('dividend_yield') else 'N/A'}
  • Beta: {fundamentals.get('beta','N/A')}
  • 52W High/Low: {curr}{fundamentals.get('week52_high','N/A')} / {curr}{fundamentals.get('week52_low','N/A')}
  • Analyst Target: {curr}{fundamentals.get('analyst_target','N/A')}
  • Analyst Recommendation: {fundamentals.get('recommendation','N/A')}
"""

    prompt = f"""
You are a senior investment analyst at a top-tier financial institution.
Your task: Generate a PROFESSIONAL, DATA-DRIVEN SWOT analysis for {name} ({ticker}).

Use ALL the data below. DO NOT be generic. Each bullet must cite specific numbers
or facts from the data provided. Be concise, high-signal, investment-grade.

{fin_block}
{price_block}
{peer_block}
RECENT NEWS (last 30 days):
{news_block}

OUTPUT FORMAT (strict — return ONLY this JSON):
{{
  "strengths": [
    {{"point": "...", "evidence": "...", "source": "...", "flag": "GREEN"}}
  ],
  "weaknesses": [
    {{"point": "...", "evidence": "...", "source": "...", "flag": "ORANGE"}}
  ],
  "opportunities": [
    {{"point": "...", "evidence": "...", "source": "...", "flag": "BLUE"}}
  ],
  "threats": [
    {{"point": "...", "evidence": "...", "source": "...", "flag": "RED"}}
  ],
  "executive_summary": "2-3 sentence investment-grade summary.",
  "investment_rating": "STRONG BUY / BUY / HOLD / SELL / STRONG SELL",
  "key_risk": "Single most critical risk in one sentence.",
  "key_opportunity": "Single most compelling opportunity in one sentence."
}}

Rules:
- Each section: minimum 4 points, maximum 7 points.
- Each point.evidence: must include actual numbers from the data (e.g. P/E, revenue, %, price).
- Each point.source: cite yfinance, news headline title, or analyst data.
- investment_rating: must be justified by the data, not generic.
- Flag meanings: GREEN=strength/positive, ORANGE=weakness/caution, BLUE=opportunity, RED=threat/risk
- Currency: use {curr} for all monetary values.
- Return ONLY valid JSON. No markdown, no explanation outside the JSON.
"""
    return prompt


def generate_swot(ticker: str, market: str,
                  fundamentals: Dict, price_metrics: Dict,
                  news: List[Dict], peers: List[Dict]) -> Dict:
    """
    Call Groq LLM with the assembled data context.
    Returns parsed SWOT dict or a structured error dict.
    """
    if not _groq_client:
        return {
            "error": "Groq API key not configured. Add GROQ_API_KEY to your .env file.",
            "fallback": True,
        }

    prompt = _build_prompt(fundamentals, price_metrics, news, peers)

    models_to_try = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant", 
        "mixtral-8x7b-32768"
    ]
    
    last_err = ""
    for model_name in models_to_try:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = _groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a senior investment analyst. "
                                "You ALWAYS respond with valid JSON only. "
                                "No markdown fences, no extra text."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=model_name,
                    temperature=0.3,
                    max_tokens=3000,
                )
                raw = response.choices[0].message.content.strip()

                # Strip markdown fences if model added them
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                swot = json.loads(raw)
                return swot

            except json.JSONDecodeError as e:
                last_err = f"LLM returned non-JSON: {e}"
                continue
            except Exception as e:
                msg = str(e)
                last_err = f"Groq error ({model_name}): {msg}"
                if "429" in msg or "Rate limit" in msg:
                    # If daily limit hit, immediately switch to next model
                    if "per day" in msg.lower() or "(tpd)" in msg.lower():
                        break
                    
                    # Otherwise (TPM limit), sleep and retry inner loop
                    if attempt < max_retries - 1:
                        time.sleep(5 + random.uniform(1.0, 3.0))
                        continue
                # For any other error, break out of retry loop and try next model
                break

    return {"error": f"Exhausted all models. Last error: {last_err}"}


# ──────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ──────────────────────────────────────────────────────────────────────

def run_swot_analysis(query: str) -> Dict:
    """
    Orchestrate full SWOT pipeline for a company name or ticker.

    Returns dict with:
      fundamentals, price_metrics, news, peers, swot
    """
    ticker, market = resolve_ticker(query)

    fundamentals = fetch_fundamentals(ticker, market)
    if "error" in fundamentals and len(fundamentals) <= 3:
        return {"error": f"Could not fetch data for '{query}' (resolved: {ticker})"}

    price_metrics = fetch_price_metrics(ticker)
    company_name = fundamentals.get("name", ticker)
    news = fetch_news(ticker, company_name)
    sector = fundamentals.get("sector", "")
    peers = fetch_peers(ticker, sector, market)
    swot = generate_swot(ticker, market, fundamentals, price_metrics, news, peers)

    return {
        "ticker": ticker,
        "market": market,
        "currency": fundamentals.get("currency", "$"),
        "company_name": company_name,
        "fundamentals": fundamentals,
        "price_metrics": price_metrics,
        "news": news,
        "peers": peers,
        "swot": swot,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
