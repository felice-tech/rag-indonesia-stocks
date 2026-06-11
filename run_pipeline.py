"""Run the full pipeline: fetch real news, train models, start API.

Usage:
    python3 run_pipeline.py        # Train + start API
    python3 run_pipeline.py --api-only   # Just start API (if models exist)
    python3 run_pipeline.py --train-only # Just train/download data
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yfinance as yf

sys.path.insert(0, ".")
from rag_geopolitik.config import DATA_DIR, load_source_registry, get_settings
from rag_geopolitik.constants import map_entities_to_tickers, ENTITY_TICKER_MAP
from rag_geopolitik.extraction import EventExtractor
from rag_geopolitik.features import FeatureBuilder, FeatureStore
from rag_geopolitik.models.flow_model import FlowModel, FLOW_FEATURES
from rag_geopolitik.models.stock_model import StockModel, STOCK_FEATURES
from rag_geopolitik.schemas import (
    RawArticle, ExtractedEvent, SentimentDirection, Magnitude, FlowDirection,
)

warnings.filterwarnings("ignore")

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# STEP 3: Real news fetcher (RSS + simple HTTP)
# ======================================================================

def _parse_rss_item(item) -> dict:
    """Extract title, link, description, pubDate from an RSS item."""
    title = item.findtext("title", "") or ""
    link = item.findtext("link", "") or ""
    desc = item.findtext("description", "") or item.findtext("content:encoded", "") or ""
    pub_str = item.findtext("pubDate", "") or ""

    pub_date = datetime.now(timezone.utc)
    if pub_str:
        try:
            from email.utils import parsedate_to_datetime
            pub_date = parsedate_to_datetime(pub_str).astimezone(timezone.utc)
        except Exception:
            pass
    return {"title": title.strip(), "link": link.strip(), "desc": desc.strip(), "pub_date": pub_date}


def _fetch_rss(url: str, source_id: str, max_articles: int) -> list[RawArticle]:
    """Generic RSS fetcher. Returns list of RawArticle."""
    import requests
    import xml.etree.ElementTree as ET
    from uuid import uuid4

    registry = load_source_registry()
    source = registry[source_id]
    articles: list[RawArticle] = []

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        try:
            content = resp.content.decode("utf-8")
        except UnicodeDecodeError:
            content = resp.content.decode("ISO-8859-1")
        root = ET.fromstring(content.encode("utf-8"))

        for item in root.findall(".//item")[:max_articles]:
            data = _parse_rss_item(item)
            if not data["title"] and not data["desc"]:
                continue
            articles.append(RawArticle(
                id=str(uuid4()),
                source_id=source_id,
                credibility_score=source.credibility_score,
                title=data["title"],
                body=data["desc"],
                url=data["link"],
                published_at=data["pub_date"],
                language=source.language,
                category="macro",
            ))
        print(f"   ✅ Fetched {len(articles)} articles from {source.name}")
    except Exception as e:
        print(f"   ⚠️  Could not fetch {source.name}: {e}")
    return articles


def fetch_kontan_news(max_articles: int = 10) -> list[RawArticle]:
    """Kontan — RSS feed."""
    return _fetch_rss("https://www.kontan.co.id/rss/latest", "kontan", max_articles)


def fetch_cnbc_news(max_articles: int = 10) -> list[RawArticle]:
    """CNBC Indonesia — RSS feed."""
    return _fetch_rss("https://www.cnbcindonesia.com/rss", "cnbc_indonesia", max_articles)


def fetch_bisnis_news(max_articles: int = 10) -> list[RawArticle]:
    """Bisnis.com — RSS feed."""
    return _fetch_rss("https://www.bisnis.com/feed.xml", "bisnis", max_articles)


def fetch_google_news_indonesia(max_articles: int = 10) -> list[RawArticle]:
    """Google News — Indonesia stock market RSS (aggregator)."""
    return _fetch_rss(
        "https://news.google.com/rss/search?q=Indonesia+stock+market+IHSG+LQ45&hl=id&gl=ID&ceid=ID:id",
        "cnbc_indonesia",  # closest match in registry
        max_articles,
    )


def fetch_all_news(max_per_source: int = 5) -> list[RawArticle]:
    """Fetch news from all available sources."""
    all_articles: list[RawArticle] = []
    # all_articles.extend(fetch_kontan_news(max_per_source))
    all_articles.extend(fetch_cnbc_news(max_per_source))
    # all_articles.extend(fetch_bisnis_news(max_per_source))
    # Fallback: Google News if no local sources worked
    if not all_articles:
        all_articles.extend(fetch_google_news_indonesia(max_per_source))
    return all_articles


# ======================================================================
# STEP 4: Train models with yfinance data
# ======================================================================

LQ45_TICKERS = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "ANTM.JK",
                 "ADRO.JK", "TLKM.JK", "UNVR.JK", "ICBP.JK", "INCO.JK",
                 "MDKA.JK", "ASII.JK", "CPIN.JK", "KLBF.JK", "PGAS.JK"]

LQ45_CLEAN = [t.replace(".JK", "") for t in LQ45_TICKERS]

# Tick symbols for proxy features
_PROXY_SYMBOLS = {
    "dxy_delta_1d": "DX-Y.NYB",       # DXY Index
    "us10y_yield_delta": "^TNX",       # 10Y Treasury Yield
    "vix_level": "^VIX",               # VIX
    "em_etf_flow": "EEM",              # Emerging Markets ETF
    "gold_price": "GC=F",              # Gold futures
    "oil_price": "CL=F",               # Crude oil
}


def download_market_data() -> dict[str, Any]:
    """Download market proxy features from Yahoo Finance."""
    print("\n📥 Downloading market data from Yahoo Finance...")
    features = {}

    for name, symbol in _PROXY_SYMBOLS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="10d")
            if len(hist) >= 2:
                close = hist["Close"]
                features[name] = float(close.iloc[-1])
                # Create delta versions
                if "delta_1d" in name or name == "vix_level":
                    features[name] = float(close.iloc[-1])
                    if len(close) >= 2:
                        features[name.replace("level", "delta_1d")] = float(
                            (close.iloc[-1] / close.iloc[-2] - 1) * 100
                        )
        except Exception as e:
            features[name] = 0.0
            print(f"   ⚠️  Could not fetch {symbol}: {e}")

    # IHSG proxy via JKSE
    try:
        ihsg = yf.Ticker("^JKSE")
        ihsg_hist = ihsg.history(period="10d")
        if len(ihsg_hist) >= 2:
            close = ihsg_hist["Close"]
            features["ihsg_return_1d"] = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
        if len(ihsg_hist) >= 5:
            features["ihsg_return_3d"] = float(
                (close.iloc[-1] / close.iloc[-4] - 1) * 100 if len(close) >= 5 else 0
            )
        if len(ihsg_hist) >= 6:
            features["ihsg_return_5d"] = float(
                (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
            )
    except Exception:
        pass

    print(f"   ✅ Market data downloaded ({len(features)} features)")
    return features


def train_flow_model_with_synthetic(target_size: int = 500) -> FlowModel:
    """Train Model A with synthetic data based on real-ish feature patterns."""
    print("\n🧠 Training Flow Model (Model A - XGBoost)...")
    np.random.seed(42)

    # Generate realistic-ish synthetic training data
    n = target_size
    X = np.zeros((n, len(FLOW_FEATURES)))

    for i, feat in enumerate(FLOW_FEATURES):
        if "delta" in feat or "return" in feat:
            X[:, i] = np.random.normal(0, 1.5, n)  # percentage changes
        elif "vix_level" in feat:
            X[:, i] = np.random.normal(18, 5, n)  # around 18
        elif "sentiment" in feat:
            X[:, i] = np.random.uniform(-0.5, 0.5, n)
        elif "idx_foreign" in feat or "em_etf" in feat:
            X[:, i] = np.random.normal(0, 500, n)
        elif "jisdor" in feat:
            X[:, i] = np.random.normal(0, 0.5, n)
        else:
            X[:, i] = np.random.uniform(0, 1, n)

    # Labels: -1 (sell), 0 (neutral), 1 (buy)
    # More buys when DXY down & sentiment positive
    dxy_col = FLOW_FEATURES.index("dxy_delta_5d") if "dxy_delta_5d" in FLOW_FEATURES else 1
    sent_col = FLOW_FEATURES.index("source_weighted_sentiment")
    prob_buy = 1 / (1 + np.exp(X[:, dxy_col] * 0.5 - X[:, sent_col] * 2))
    y = np.array([np.random.choice([-1, 0, 1], p=[0.2, 0.3, 0.5]) if p > 0.5
                  else np.random.choice([-1, 0, 1], p=[0.5, 0.3, 0.2])
                  for p in prob_buy])

    model = FlowModel()
    model.train(X, y)
    model.save(MODEL_DIR / "flow_model.pkl")
    print(f"   ✅ Flow Model trained with {n} samples, saved to {MODEL_DIR / 'flow_model.pkl'}")
    return model


def train_stock_model_with_synthetic(target_size: int = 2000) -> StockModel:
    """Train Model B with synthetic data."""
    print("\n🧠 Training Stock Model (Model B - LightGBM)...")
    np.random.seed(42)

    n = target_size
    X = np.zeros((n, len(STOCK_FEATURES)))

    for i, feat in enumerate(STOCK_FEATURES):
        if "return" in feat or "delta" in feat:
            X[:, i] = np.random.normal(0, 2, n)
        elif "confidence" in feat or "probability" in feat:
            X[:, i] = np.random.uniform(0.3, 0.8, n)
        elif "volume" in feat:
            X[:, i] = np.random.lognormal(0, 0.5, n)
        elif "score" in feat or "level" in feat:
            X[:, i] = np.random.uniform(0, 1, n)
        elif "velocity" in feat:
            X[:, i] = np.random.poisson(5, n)
        else:
            X[:, i] = np.random.uniform(-1, 1, n)

    # Binary labels: 1 = outperform, 0 = underperform
    # More outperform when market momentum + sentiment are positive
    momentum = X[:, STOCK_FEATURES.index("ihsg_return_5d")]
    sentiment = X[:, STOCK_FEATURES.index("ticker_sentiment_score")]
    prob = 1 / (1 + np.exp(-momentum * 0.5 - sentiment * 2))
    y = np.random.binomial(1, prob)

    model = StockModel()
    model.train(X, y)
    model.save(MODEL_DIR / "stock_model.pkl")
    print(f"   ✅ Stock Model trained with {n} samples, saved to {MODEL_DIR / 'stock_model.pkl'}")
    return model


# ======================================================================
# Per-ticker feature download from Yahoo Finance
# ======================================================================

def download_ticker_features(ticker: str, flow_direction_val: float, flow_confidence: float) -> dict[str, Any]:
    """Download real per-ticker features from Yahoo Finance."""
    import yfinance as yf
    feats: dict[str, Any] = {}

    try:
        t = yf.Ticker(f"{ticker}.JK")
        hist = t.history(period="10d")
        if len(hist) >= 2:
            close = hist["Close"]
            feats["ticker_return_1d"] = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
        if len(hist) >= 6:
            feats["ticker_return_5d"] = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)

        # Actual volume (shares traded) and total accumulation (volume × price)
        if len(hist) >= 1:
            last_close = float(hist["Close"].iloc[-1])
            last_vol = int(hist["Volume"].iloc[-1])
            feats["ticker_volume"] = last_vol
            # Total accumulation = shares × price (in billions)
            feats["total_accumulation"] = round(last_vol * last_close / 1_000_000_000, 2)
    except Exception:
        pass

    # Defaults when Yahoo data unavailable
    feats.setdefault("ticker_return_1d", 0.0)
    feats.setdefault("ticker_return_5d", 0.0)
    feats.setdefault("ticker_volume", 0)
    feats.setdefault("total_accumulation", 0.0)

    # These require NLP pipeline — set defaults for demo
    feats["ticker_sentiment_score"] = 0.0
    feats["ticker_event_velocity"] = 0
    feats["novelty_score"] = 0.5
    feats["sector_exposure_score"] = 0.5
    feats["relevant_commodity_delta"] = 0.0
    feats["predicted_flow_direction"] = flow_direction_val
    feats["predicted_flow_confidence"] = flow_confidence

    return feats


# ======================================================================
# STEP 5: Start API
# ======================================================================

def start_api(flow_model: FlowModel, stock_model: StockModel):
    """Launch FastAPI server."""
    from rag_geopolitik.api import create_app
    from rag_geopolitik.recommendation import Recommender
    import uvicorn

    # Wire up a recommender with the stock model
    recommender = Recommender(stock_model=stock_model)

    app = create_app(
        flow_model=flow_model,
        stock_model=stock_model,
        recommender=recommender,
    )

    print("\n" + "=" * 60)
    print("🚀 API Server starting at http://localhost:6000")
    print("📖 Swagger docs: http://localhost:6000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=6000)


# ======================================================================
# Quick prediction demo
# ======================================================================

def show_predictions(flow_model: FlowModel, stock_model: StockModel):
    """Run predictions on a few sample tickers and print results."""
    print("\n" + "=" * 60)
    print("📊 PREDICTIONS")
    print("=" * 60)

    # 1. Market-level flow prediction
    market_feats = download_market_data()
    flow_pred = flow_model.predict(market_feats)
    print(f"\n🌊 Foreign Flow: {flow_pred.direction.value.upper()}")
    print(f"   Confidence: {flow_pred.confidence:.0%}")
    print(f"   Est. Value: {flow_pred.estimated_value}")

    # 2. Foreign Flow: top net buy / net sell by return with volume & total accumulation
    flow_dir_val = 1.0 if flow_pred.direction in (FlowDirection.NET_BUY, FlowDirection.NEUTRAL) else -1.0
    all_ticker_data: list[tuple[str, float, int, float]] = []
    for ticker in LQ45_CLEAN:
        tf = download_ticker_features(ticker, flow_dir_val, flow_pred.confidence)
        ret  = tf.get("ticker_return_1d", 0.0)
        shrs = tf.get("ticker_volume", 0)
        accum= tf.get("total_accumulation", 0.0)
        all_ticker_data.append((ticker, ret, shrs, accum))

    buys  = sorted([t for t in all_ticker_data if t[1] >= 0], key=lambda x: x[1], reverse=True)[:5]
    sells = sorted([t for t in all_ticker_data if t[1] <  0], key=lambda x: x[1])[:5]

    def fmt_shares(n: int) -> str:
        """Format shares: e.g. 12_500_000 -> '12.5M' or 1_200_000_000 -> '1.2B'"""
        if n >= 1_000_000_000:
            return f"{n/1_000_000_000:.1f}B"
        return f"{n/1_000_000:.1f}M"

    print(f"\n📊 Foreign Flow Activity — today (real Yahoo Finance data):")
    print(f"   {'─'*70}")
    print(f"   🔵 TOP 5 NET BUY (by today's return):")
    for ticker, ret, shrs, accum in buys:
        print(f"      {ticker:5s}  return={ret:+.2f}%  shares={fmt_shares(shrs):>8s}  accum=IDR{accum:.2f}Bn")
    print(f"   {'─'*70}")
    print(f"   🔴 TOP 5 NET SELL (by today's return):")
    for ticker, ret, shrs, accum in sells:
        print(f"      {ticker:5s}  return={ret:+.2f}%  shares={fmt_shares(shrs):>8s}  accum=IDR{accum:.2f}Bn")
    print(f"   {'─'*70}")

    # 3. Per-ticker predictions with explanations
    print(f"\n📈 Top stock opportunities (ranked by outperform probability):")
    print(f"   🌐 Market context: Flow={flow_pred.direction.value}, Confidence={flow_pred.confidence:.0%}")
    opportunities = []

    for ticker in LQ45_CLEAN:
        flow_dir_val = 1.0 if flow_pred.direction in (FlowDirection.NET_BUY, FlowDirection.NEUTRAL) else -1.0
        ticker_feats = download_ticker_features(ticker, flow_dir_val, flow_pred.confidence)
        feats = {**market_feats, **ticker_feats, "ticker": ticker}
        # Fill default zeros for missing features
        for f in STOCK_FEATURES:
            feats.setdefault(f, 0.0)

        opp = stock_model.predict(ticker, feats)
        # Build explanation summary
        drivers: list[tuple[str, float]] = [
            ("sentiment", feats["ticker_sentiment_score"]),
            ("return_1d", feats["ticker_return_1d"]),
            ("volume", feats["ticker_volume_ratio"]),
            ("events", feats["ticker_event_velocity"]),
        ]
        # Show what helped (+) or hurt (-) the most
        pos = [f"{n}={v:+.2f}" for n, v in drivers if v > 0]
        neg = [f"{n}={v:.2f}" for n, v in drivers if v < 0]
        why_parts = []
        if pos:
            why_parts.append(f"boosted by: {', '.join(pos[:2])}")
        if neg:
            why_parts.append(f"weighed by: {', '.join(neg[:2])}")
        explanation = " | ".join(why_parts) if why_parts else "neutral signals"

        opportunities.append((ticker, opp.outperform_probability, opp.confidence, feats, explanation))

    opportunities.sort(key=lambda x: x[1], reverse=True)
    print(f"\n   {'Ticker':5s}  {'Prob':5s}  {'Conf':5s}  Why")
    print(f"   {'-'*5}  {'-'*5}  {'-'*5}  {'-'*55}")
    for ticker, prob, conf, feats, explanation in opportunities:
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"   {ticker:5s}  {bar}  {prob:.0%}  {conf:.0%}  {explanation}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Geopolitik Pipeline")
    parser.add_argument("--api-only", action="store_true", help="Skip training, just start API")
    parser.add_argument("--train-only", action="store_true", help="Just train and save models")
    args = parser.parse_args()

    print("=" * 60)
    print("🏭 RAG Geopolitik & Investasi — Pipeline")
    print("=" * 60)

    flow_model = None
    stock_model = None

    # Try loading saved models
    flow_path = MODEL_DIR / "flow_model.pkl"
    stock_path = MODEL_DIR / "stock_model.pkl"

    if args.api_only:
        # Load saved models
        if flow_path.exists():
            flow_model = FlowModel(model_path=flow_path)
            print("   ✅ Loaded Flow Model from disk")
        else:
            print("   ⚠️  No Flow Model found, training new one...")
            flow_model = train_flow_model_with_synthetic()

        if stock_path.exists():
            stock_model = StockModel(model_path=stock_path)
            print("   ✅ Loaded Stock Model from disk")
        else:
            print("   ⚠️  No Stock Model found, training new one...")
            stock_model = train_stock_model_with_synthetic()
    else:
        # Step 3: Fetch news from all sources
        print("\n📡 Step 3: Fetching real news from all sources...")
        articles = fetch_all_news(max_per_source=5)
        print(f"\n   📊 Total articles fetched: {len(articles)}")
        if articles:
            extractor = EventExtractor()
            events = extractor.extract_batch(articles)
            print(f"   Extracted {len(events)} events from news")
            for e in events:
                print(f"   - [{e.event_type:15s}] sentiment={e.sentiment_score:+.2f}  cred={e.source_credibility:.2f}")
        else:
            events = []

        # Step 4: Train models
        print("\n🧠 Step 4: Training models...")
        flow_model = train_flow_model_with_synthetic()
        stock_model = train_stock_model_with_synthetic()

        # Show predictions
        show_predictions(flow_model, stock_model)

        if args.train_only:
            print("\n✅ Models trained and saved. Skipping API start.")
            return

    # Step 5: Start API
    print("\n🚀 Step 5: Starting API server...")
    start_api(flow_model, stock_model)


if __name__ == "__main__":
    main()