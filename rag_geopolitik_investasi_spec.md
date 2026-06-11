# RAG Geopolitik & Investasi — Full Technical Specification
> Scope: LQ45 | Output: Foreign Flow Prediction + Stock Opportunity | Stack: Python

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [News Sources & Collector](#2-news-sources--collector)
3. [RAG Knowledge Base](#3-rag-knowledge-base)
4. [Event Extraction](#4-event-extraction)
5. [Feature Engineering](#5-feature-engineering)
6. [Feature Store](#6-feature-store)
7. [ML Prediction Engine](#7-ml-prediction-engine)
8. [Recommendation Engine](#8-recommendation-engine)
9. [Dashboard / API](#9-dashboard--api)
10. [Foreign Flow Data Strategy](#10-foreign-flow-data-strategy)
11. [Output Schema](#11-output-schema)
12. [Tech Stack Summary](#12-tech-stack-summary)

---

## 1. System Overview

```
News Sources
    ↓ (crawl + source score)
News Collector
    ↓ (dedup, normalize)
RAG Knowledge Base
    ↓ (weighted retrieval)
Event Extraction
    ↓ (NER, classifier, sentiment)
Feature Engineering
    ↓ (4 feature buckets)
Feature Store
    ↓ (real-time + historical)
ML Prediction Engine
    ↓ (XGBoost + LightGBM)
Recommendation Engine
    ↓ (event matching + explanation)
Dashboard / API
```

### Hard Rules
- Setiap news source memiliki **credibility score (0.0–1.0)**
- Score ini menjadi weight di retrieval RAG dan di feature input
- Scope saham: **LQ45 only**
- Semua output harus disertai **confidence score** dan **explanation**

---

## 2. News Sources & Collector

### 2.1 Source Registry

| Source | Type | Language | Credibility Score | Notes |
|---|---|---|---|---|
| Reuters | Wire | EN | 0.95 | Prioritas utama macro global |
| Bloomberg | Wire | EN | 0.93 | Fokus market & commodities |
| CNBC Indonesia | Portal | ID | 0.80 | Lokal, cepat |
| Kontan | Portal | ID | 0.82 | Fokus pasar modal IDX |
| Bisnis.com | Portal | ID | 0.78 | Good for earnings news |
| Twitter/X (verified) | Social | ID/EN | 0.55 | Hanya akun verified analyst |
| Telegram channel | Social | ID | 0.50 | Whitelist channel saja |

### 2.2 Crawler Architecture

```python
# Tech: Scrapy + Playwright (untuk JS-rendered pages)
# Schedule: setiap 30 menit (cron / APScheduler)

class NewsCrawler:
    def __init__(self):
        self.source_registry = load_source_registry()   # YAML config
        self.dedup_store = RedisDedup()                  # hash judul+url

    def crawl(self, source: Source) -> list[RawArticle]:
        # 1. Fetch HTML / JSON API
        # 2. Parse title, body, published_at, url, author
        # 3. Dedup check (SHA256 dari title+url)
        # 4. Normalize timestamp → UTC
        # 5. Attach source metadata (credibility_score, category)
        pass
```

### 2.3 Article Schema

```python
@dataclass
class RawArticle:
    id: str                    # UUID
    source_id: str             # "reuters", "kontan", dll
    credibility_score: float   # dari source registry
    title: str
    body: str
    url: str
    published_at: datetime     # UTC
    language: str              # "id" atau "en"
    category: str              # "macro", "commodity", "earnings", "geopolitical"
    raw_html: str | None
```

### 2.4 Deduplication Strategy

- Hash `SHA256(title + source_id)` → cek di Redis dengan TTL 7 hari
- Fuzzy dedup untuk artikel yang sama dari sumber berbeda: `difflib.SequenceMatcher ratio > 0.85` → keep yang `credibility_score` lebih tinggi

---

## 3. RAG Knowledge Base

### 3.1 Stack

| Komponen | Pilihan | Alasan |
|---|---|---|
| Vector DB | Qdrant | Support payload filtering, mudah filter by `credibility_score` |
| Embedding model | `text-embedding-3-small` (OpenAI) | Cost-efficient, multilingual |
| Chunking | 512 token, overlap 64 token | Balance antara context dan precision |

### 3.2 Ingestion Pipeline

```python
def ingest_article(article: RawArticle):
    # 1. Chunking
    chunks = chunk_text(article.body, size=512, overlap=64)

    # 2. Embedding
    embeddings = openai_embed(chunks)

    # 3. Store ke Qdrant dengan metadata
    for chunk, embedding in zip(chunks, embeddings):
        qdrant.upsert(
            collection="news_kb",
            points=[{
                "id": uuid4(),
                "vector": embedding,
                "payload": {
                    "article_id": article.id,
                    "source_id": article.source_id,
                    "credibility_score": article.credibility_score,
                    "published_at": article.published_at.isoformat(),
                    "category": article.category,
                    "chunk_text": chunk,
                }
            }]
        )
```

### 3.3 Weighted Retrieval

Final relevance score menggabungkan semantic similarity dengan source credibility:

```python
def weighted_retrieve(query: str, top_k: int = 10) -> list[Chunk]:
    raw_results = qdrant.search(
        collection="news_kb",
        query_vector=openai_embed(query),
        limit=top_k * 2,          # ambil lebih, lalu re-rank
        query_filter={"published_at": {"gte": 7_days_ago}}  # max 7 hari
    )

    # Re-rank dengan source weight
    for r in raw_results:
        r.final_score = (
            0.7 * r.score +                          # cosine similarity
            0.3 * r.payload["credibility_score"]     # source credibility
        )

    return sorted(raw_results, key=lambda x: x.final_score, reverse=True)[:top_k]
```

---

## 4. Event Extraction

### 4.1 Pipeline

```
Raw Article
    → Language Detection (langdetect)
    → Translation if needed (EN → ID, DeepL API)
    → NER (spaCy + custom model)
    → Event Type Classification
    → Sentiment Analysis
    → Entity-to-Ticker Linking
    → Structured Event Output
```

### 4.2 Event Type Taxonomy

```python
EVENT_TYPES = {
    "macro": [
        "rate_decision",        # Fed, BI rate
        "gdp_release",
        "inflation_data",
        "trade_balance",
    ],
    "geopolitical": [
        "trade_war",
        "sanctions",
        "conflict",
        "election",
        "diplomatic_tension",
    ],
    "commodity": [
        "commodity_price_shock",  # nickel, CPO, coal, oil
        "supply_disruption",
        "demand_outlook_change",
    ],
    "corporate": [
        "earnings_beat",
        "earnings_miss",
        "dividend_announcement",
        "rights_issue",
        "merger_acquisition",
    ],
    "capital_flow": [
        "foreign_inflow_signal",
        "foreign_outflow_signal",
        "em_risk_on",
        "em_risk_off",
    ]
}
```

### 4.3 Entity-to-Ticker Mapping

```python
# Contoh mapping (diperluas untuk semua LQ45)
ENTITY_TICKER_MAP = {
    # Commodity exposure
    "nickel": ["ANTM", "INCO", "MDKA"],
    "cpo": ["AALI", "LSIP", "SIMP"],
    "coal": ["ADRO", "PTBA", "ITMG", "BUMI"],
    "gold": ["ANTM", "MDKA"],
    "crude oil": ["MEDC", "ELSA"],

    # Macro / sector exposure
    "bank indonesia": ["BBCA", "BBRI", "BMRI", "BBNI"],
    "fed rate": ["BBCA", "BBRI", "BMRI", "BBNI"],  # bank sensitif
    "china stimulus": ["ANTM", "INCO", "ADRO"],    # commodity demand
    "usd idr": ["BBCA", "BBRI", "UNVR", "ICBP"],  # FX sensitive

    # Direct mention
    "bank central asia": ["BBCA"],
    "bca": ["BBCA"],
    "telkom": ["TLKM"],
    # ... dst
}
```

### 4.4 Structured Event Schema

```python
@dataclass
class ExtractedEvent:
    event_id: str
    article_id: str
    event_type: str              # dari EVENT_TYPES
    entities: list[str]          # nama entitas yang ditemukan
    tickers_affected: list[str]  # hasil entity-to-ticker mapping
    sentiment_score: float       # -1.0 (bearish) → +1.0 (bullish)
    sentiment_direction: str     # "bullish", "bearish", "neutral"
    magnitude: str               # "high", "medium", "low"
    novelty_score: float         # 0–1, seberapa baru vs historical events
    published_at: datetime
    source_credibility: float
    raw_text_snippet: str        # kalimat kunci yang jadi basis ekstraksi
```

---

## 5. Feature Engineering

### 5.1 Feature Buckets Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Feature Store                          │
├──────────────┬──────────────┬──────────────┬───────────────┤
│   Macro /    │   Sector /   │   Market     │   Foreign     │
│  Geopolitic  │  Commodity   │ Microstructure│   Flow Proxy  │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

### 5.2 Bucket 1: Macro / Geopolitical Features

| Feature | Tipe | Deskripsi |
|---|---|---|
| `event_type_encoded` | categorical (one-hot) | Jenis event dari taxonomy |
| `sentiment_score` | float [-1, 1] | Weighted average dari semua berita hari ini |
| `source_weighted_sentiment` | float [-1, 1] | Sentiment × credibility_score |
| `event_velocity` | int | Jumlah artikel serupa dalam 24 jam terakhir |
| `novelty_score` | float [0, 1] | Cosine distance vs historical events serupa |
| `country_pair_china_us` | bool | Ada tensi China-US? |
| `country_pair_id_us` | bool | Ada isu bilateral RI-AS? |
| `geopolitical_risk_index` | float | Composite dari event severity × velocity |

### 5.3 Bucket 2: Sector / Commodity Features

| Feature | Tipe | Sumber Data |
|---|---|---|
| `nickel_price_delta_1d` | float % | LME Nickel (Yahoo Finance) |
| `nickel_price_delta_5d` | float % | LME Nickel |
| `cpo_price_delta_1d` | float % | CME Palm Oil |
| `coal_price_delta_1d` | float % | ICE Newcastle Coal |
| `gold_price_delta_1d` | float % | COMEX Gold |
| `oil_price_delta_1d` | float % | Brent Crude |
| `commodity_sentiment` | float [-1, 1] | Dari berita commodity |
| `sector_exposure_score` | float [0, 1] | Seberapa besar ticker terekspos ke commodity event |

### 5.4 Bucket 3: Market Microstructure Features

| Feature | Tipe | Deskripsi |
|---|---|---|
| `ihsg_return_1d` | float % | IHSG daily return |
| `ihsg_return_3d` | float % | IHSG 3-day return |
| `ihsg_return_5d` | float % | IHSG 5-day return |
| `ihsg_volatility_10d` | float | Rolling 10-day std dev |
| `lq45_vs_ihsg_rs` | float | Relative strength LQ45 vs IHSG |
| `ticker_return_1d` | float % | Per-ticker daily return |
| `ticker_return_5d` | float % | Per-ticker 5-day return |
| `ticker_volume_ratio` | float | Volume hari ini / avg 20 hari |
| `earnings_season` | bool | Apakah sedang musim laporan keuangan |
| `quarter` | int [1–4] | Kuartal berjalan |

### 5.5 Bucket 4: Foreign Flow Proxy Features

| Feature | Tipe | Deskripsi | Sumber |
|---|---|---|---|
| `dxy_delta_1d` | float % | DXY index change | Yahoo Finance |
| `dxy_delta_5d` | float % | DXY 5-day change | Yahoo Finance |
| `us10y_yield_delta` | float bps | Perubahan yield US 10Y | Yahoo Finance |
| `vix_level` | float | Fear index | Yahoo Finance |
| `vix_delta_1d` | float % | Perubahan VIX | Yahoo Finance |
| `em_etf_flow` | float | EEM ETF flow proxy | Yahoo Finance (EEM) |
| `jisdor_delta` | float % | Perubahan kurs JISDOR | BI website scraping |
| `idx_foreign_net_buy` | float (IDR) | Net foreign buy/sell (delay 1 hari) | IDX scraping |
| `idx_foreign_net_buy_5d` | float (IDR) | Rolling 5-day net buy | IDX scraping |
| `broker_flow_estimate` | float | Estimasi dari broker report | Mandiri/RHB PDF parsing |

### 5.6 Feature Importance Baseline (SHAP expected ranking)

Berdasarkan literatur dan karakteristik pasar IDX:

```
1. dxy_delta_5d              → paling predictive untuk foreign flow direction
2. us10y_yield_delta         → capital flight indicator
3. vix_level                 → risk-on / risk-off signal
4. idx_foreign_net_buy_5d    → lagged actual flow (delay 1 hari)
5. ihsg_return_3d            → momentum
6. source_weighted_sentiment → news signal
7. commodity_price_delta_1d  → sector-specific
8. novelty_score             → menghindari priced-in events
```

---

## 6. Feature Store

### 6.1 Arsitektur

```
                    ┌─────────────────┐
  Real-time data ──►│   Redis Cache   │◄── API request (< 100ms)
                    └────────┬────────┘
                             │ sync setiap malam
                    ┌────────▼────────┐
  Historical data ──►│ Parquet files  │◄── ML training / backtesting
                    │  (S3 / lokal)  │
                    └─────────────────┘
```

### 6.2 Schema Redis Key

```
feature:{ticker}:{date}        → JSON feature vector per ticker per hari
feature:market:{date}          → JSON market-wide features (IHSG, DXY, VIX)
feature:flow:{date}            → JSON foreign flow features
event:latest                   → List of ExtractedEvent 24 jam terakhir
```

### 6.3 Parquet Schema (untuk training)

```python
# Satu row = satu ticker, satu tanggal
schema = {
    "date": "date",
    "ticker": "string",
    # Macro features
    "sentiment_score": "float32",
    "source_weighted_sentiment": "float32",
    "event_velocity": "int32",
    "novelty_score": "float32",
    "geopolitical_risk_index": "float32",
    # Commodity features
    "nickel_price_delta_1d": "float32",
    "cpo_price_delta_1d": "float32",
    "coal_price_delta_1d": "float32",
    # Market features
    "ihsg_return_1d": "float32",
    "ihsg_return_5d": "float32",
    "ticker_return_1d": "float32",
    "ticker_volume_ratio": "float32",
    # Flow proxy features
    "dxy_delta_1d": "float32",
    "dxy_delta_5d": "float32",
    "us10y_yield_delta": "float32",
    "vix_level": "float32",
    "idx_foreign_net_buy_5d": "float32",
    # Target labels (untuk training)
    "target_foreign_flow_direction": "int8",    # -1, 0, 1
    "target_ticker_outperform_5d": "float32",  # return ticker - return IHSG dalam 5 hari
    "target_ticker_outperform_bool": "bool",   # True jika > +2%
}
```

---

## 7. ML Prediction Engine

### 7.1 Model A — Foreign Flow Direction Classifier

**Tujuan**: Prediksi arah net foreign buy/sell di pasar IDX secara keseluruhan

```python
# Model: XGBoost Classifier
# Output: {-1: net_sell, 0: neutral, 1: net_buy}
# Granularitas: market-level (bukan per ticker)

from xgboost import XGBClassifier

model_a = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)

# Features yang digunakan
FLOW_FEATURES = [
    "dxy_delta_1d", "dxy_delta_5d",
    "us10y_yield_delta",
    "vix_level", "vix_delta_1d",
    "em_etf_flow",
    "jisdor_delta",
    "ihsg_return_3d",
    "source_weighted_sentiment",
    "geopolitical_risk_index",
    "idx_foreign_net_buy_5d",   # lagged
]
```

### 7.2 Model B — Stock Outperform Probability

**Tujuan**: Prediksi probabilitas ticker LQ45 outperform IHSG dalam 5 hari ke depan

```python
# Model: LightGBM
# Output: float [0, 1] → probabilitas outperform
# Granularitas: per ticker

import lightgbm as lgb

model_b = lgb.LGBMClassifier(
    n_estimators=500,
    num_leaves=31,
    learning_rate=0.03,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    random_state=42
)

# Features yang digunakan (gabungan semua bucket + output Model A)
STOCK_FEATURES = [
    # Dari Model A output
    "predicted_flow_direction",
    "predicted_flow_confidence",
    # Market features
    "ihsg_return_1d", "ihsg_return_5d",
    "ticker_return_1d", "ticker_return_5d",
    "ticker_volume_ratio",
    # Commodity (ticker-specific weight)
    "sector_exposure_score",
    "relevant_commodity_delta",
    # News features
    "ticker_sentiment_score",
    "ticker_event_velocity",
    "novelty_score",
    # Flow proxy
    "dxy_delta_5d",
    "vix_level",
]
```

### 7.3 Training Protocol

```python
# Time-series split (WAJIB, jangan random split)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=5)  # gap 5 hari untuk avoid leakage

# Walk-forward validation
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

### 7.4 Evaluation Metrics

| Model | Primary Metric | Secondary |
|---|---|---|
| Model A (flow direction) | Accuracy + F1 (weighted) | Confusion matrix per class |
| Model B (stock outperform) | AUC-ROC | Precision@Top10 (top 10 predicted vs actual outperformers) |

### 7.5 SHAP Explainability

```python
import shap

explainer_a = shap.TreeExplainer(model_a)
explainer_b = shap.TreeExplainer(model_b)

def get_shap_explanation(ticker: str, features: dict) -> dict:
    shap_values = explainer_b.shap_values(features)
    # Ambil top 5 feature dengan absolute SHAP value terbesar
    top_features = sorted(
        zip(STOCK_FEATURES, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]
    return {name: value for name, value in top_features}
```

---

## 8. Recommendation Engine

### 8.1 Pipeline

```
Model A Output (flow direction + confidence)
Model B Output (outperform_prob per ticker)
    ↓
Historical Event Matching (RAG retrieval)
    ↓
Novelty Score Check (jika < 0.3 → priced-in warning)
    ↓
Confidence Adjuster
    ↓
Explanation Generator (LLM via RAG)
    ↓
Final JSON Output
```

### 8.2 Historical Event Matching

```python
def match_historical_events(current_event: ExtractedEvent) -> list[HistoricalEvent]:
    """
    Cari event serupa di masa lalu dan lihat dampaknya ke saham
    """
    query = f"{current_event.event_type} {' '.join(current_event.entities)}"
    similar_events = qdrant.search(
        collection="historical_events",  # collection terpisah
        query_vector=openai_embed(query),
        limit=5,
        query_filter={"published_at": {"lt": today}}
    )
    # Return event + actual stock return setelahnya (5 hari)
    return enrich_with_historical_returns(similar_events)
```

### 8.3 Confidence Adjuster Logic

```python
def adjust_confidence(
    model_confidence: float,
    novelty_score: float,
    source_credibility_avg: float,
    n_sources_confirming: int,
) -> float:
    """
    Confidence turun jika:
    - novelty_score rendah (event sudah priced-in)
    - source credibility rendah
    - hanya sedikit sumber yang mengkonfirmasi
    """
    novelty_penalty = (1 - novelty_score) * 0.2    # max -20%
    source_boost = (source_credibility_avg - 0.5) * 0.1
    confirmation_boost = min(n_sources_confirming / 5, 1) * 0.1

    adjusted = model_confidence - novelty_penalty + source_boost + confirmation_boost
    return round(min(max(adjusted, 0.1), 0.99), 2)
```

### 8.4 Explanation Generator

```python
def generate_explanation(
    ticker: str,
    shap_features: dict,
    historical_matches: list,
    flow_direction: str,
) -> str:
    """
    Gunakan LLM untuk generate human-readable explanation
    dari SHAP values + historical context
    """
    context = weighted_retrieve(
        query=f"{ticker} outlook {' '.join(shap_features.keys())}",
        top_k=5
    )

    prompt = f"""
    Berdasarkan analisis berita dan model prediksi, jelaskan mengapa {ticker}
    memiliki potensi outperform dalam 5 hari ke depan.

    Top driving factors (dari SHAP):
    {shap_features}

    Historical event serupa:
    {historical_matches}

    Prediksi foreign flow: {flow_direction}

    Konteks berita terkini:
    {[c.chunk_text for c in context]}

    Berikan penjelasan dalam format bullet point, maksimal 5 poin, bahasa Indonesia.
    Sertakan data historis jika relevan (misal: 'event serupa menghasilkan +X%').
    """

    return call_llm(prompt)
```

---

## 9. Dashboard / API

### 9.1 FastAPI Endpoints

```python
# GET /api/v1/prediction/daily
# Mengembalikan semua prediksi untuk hari ini

# GET /api/v1/prediction/ticker/{ticker}
# Detail prediksi per ticker

# GET /api/v1/foreign-flow/prediction
# Foreign flow prediction market-level

# POST /api/v1/news/ingest
# Manual ingest artikel (untuk testing)

# GET /api/v1/features/{ticker}/{date}
# Lihat feature vector yang digunakan

# GET /api/v1/explanation/{ticker}
# Full explanation + SHAP chart data
```

### 9.2 Telegram Alert Bot

```python
# Alert dikirim ketika:
# - outperform_probability > 75% AND confidence > 70%
# - foreign flow reversal signal (dari sell ke buy atau sebaliknya)
# - novelty_score tinggi (event benar-benar baru)

ALERT_TEMPLATE = """
🔔 *Signal Alert — {ticker}*

📊 Outperform Prob: {prob}%
🎯 Confidence: {confidence}%
🌊 Foreign Flow: {flow_direction}

*Alasan:*
{explanation_bullets}

⚠️ _Bukan rekomendasi investasi. DYOR._
"""
```

---

## 10. Foreign Flow Data Strategy

### 10.1 Tier Prioritas

| Tier | Sumber | Delay | Granularitas | Biaya |
|---|---|---|---|---|
| 1 | IDX JATS scraping | 1 hari | Per ticker | Gratis |
| 2 | Stockbit / RTI API | Real-time | Per ticker | Berbayar |
| 3 | DXY + US10Y proxy | Real-time | Market level | Gratis |
| 4 | Broker weekly report | 1 minggu | Sektoral | Gratis |

### 10.2 IDX Daily Foreign Flow Scraper

```python
# Source: https://www.idx.co.id/en/market-data/stock-data/foreign-net-buy-sell/
# Format: tabel HTML, update H+1 pukul 09:00

def scrape_idx_foreign_flow(date: str) -> dict[str, float]:
    """
    Return: {"BBCA": 230_000_000_000, "ANTM": -45_000_000_000, ...}
    Satuan: IDR
    """
    url = f"https://www.idx.co.id/en/market-data/stock-data/foreign-net-buy-sell/?date={date}"
    # Parse tabel dengan pandas.read_html atau BeautifulSoup
    pass
```

### 10.3 Proxy Model (ketika data aktual tidak tersedia)

```python
# Regresi sederhana untuk estimasi flow direction dari proxy variables
# Di-train dari historical IDX foreign flow + DXY + US10Y + VIX

PROXY_EQUATION = """
flow_direction ≈
    -0.45 × dxy_delta_5d
    -0.30 × us10y_yield_delta
    -0.15 × vix_delta_1d
    +0.10 × ihsg_return_3d
"""
```

---

## 11. Output Schema

### 11.1 Foreign Flow Prediction

```json
{
  "date": "2025-06-09",
  "market_flow": {
    "direction": "net_buy",
    "estimated_value": "+1.2T",
    "confidence": 74,
    "driving_factors": ["DXY -0.8%", "US10Y turun 5bps", "risk-on sentiment"]
  },
  "by_sector": {
    "banking": {"direction": "net_buy", "confidence": 80},
    "mining": {"direction": "net_buy", "confidence": 72},
    "consumer": {"direction": "neutral", "confidence": 55}
  },
  "per_ticker": {
    "BBCA": {
      "foreign_flow": "+230B",
      "confidence": 82
    },
    "ANTM": {
      "foreign_flow": "+45B",
      "confidence": 68
    }
  }
}
```

### 11.2 Stock Opportunity

```json
{
  "date": "2025-06-09",
  "opportunities": [
    {
      "ticker": "ANTM",
      "outperform_probability": 78,
      "confidence": 81,
      "horizon_days": 5,
      "explanation": {
        "summary": "ANTM bullish karena kombinasi China stimulus dan foreign inflow",
        "bullets": [
          "China stimulus → demand nickel meningkat",
          "Nickel LME +2.3% dalam 3 hari terakhir",
          "Foreign inflow expected (DXY -0.8%)",
          "Event serupa pada Okt 2023 menghasilkan +4.2% dalam 5 hari",
          "Volume ANTM 1.8x rata-rata 20 hari → akumulasi terdeteksi"
        ],
        "top_shap_features": {
          "nickel_price_delta_1d": 0.18,
          "dxy_delta_5d": -0.15,
          "ticker_volume_ratio": 0.12,
          "source_weighted_sentiment": 0.10,
          "novelty_score": 0.08
        },
        "historical_reference": {
          "event": "China PMI beat + stimulus Nov 2023",
          "antm_return_5d": "+4.2%",
          "similarity_score": 0.84
        }
      },
      "risk_flags": [],
      "novelty_score": 0.72
    }
  ]
}
```

### 11.3 Risk Flags

```python
RISK_FLAGS = {
    "low_novelty": "Event kemungkinan sudah priced-in (novelty < 0.4)",
    "single_source": "Hanya 1 sumber mengkonfirmasi, perlu validasi",
    "earnings_window": "Laporan keuangan akan rilis dalam 5 hari",
    "high_vix": "VIX > 25, market sedang fear mode",
    "low_liquidity": "Volume rata-rata rendah, prediksi kurang reliable",
}
```

---

## 12. Tech Stack Summary

| Layer | Library / Tool | Versi |
|---|---|---|
| Web scraping | Scrapy + Playwright | Scrapy 2.11, Playwright 1.44 |
| NLP / NER | spaCy + transformers | spaCy 3.7 |
| Embedding | OpenAI text-embedding-3-small | via API |
| Vector DB | Qdrant | 1.9.x |
| Feature store (cache) | Redis | 7.x |
| Feature store (historical) | Parquet + pandas | pandas 2.x |
| ML (flow) | XGBoost | 2.0.x |
| ML (stock) | LightGBM | 4.x |
| Explainability | SHAP | 0.45.x |
| LLM (explanation) | Claude claude-sonnet-4-20250514 | via API |
| API | FastAPI | 0.111.x |
| Dashboard | Streamlit | 1.35.x |
| Scheduler | APScheduler | 3.10.x |
| Alert | python-telegram-bot | 21.x |
| Containerization | Docker + Docker Compose | - |

---

## Appendix: Development Phases

### Phase 1 — Foundation (2–3 minggu)
- [ ] Setup crawler untuk 3 sumber utama (Reuters, Kontan, CNBC ID)
- [ ] RAG knowledge base dengan Qdrant
- [ ] Basic event extraction (rule-based dulu, ML kemudian)
- [ ] Historical data collection: IHSG, LQ45, commodity prices

### Phase 2 — Feature Engineering (2 minggu)
- [ ] Implement semua 4 feature buckets
- [ ] IDX foreign flow scraper
- [ ] Feature store dengan Redis + Parquet
- [ ] Validasi korelasi features vs actual stock movement

### Phase 3 — ML Models (3 minggu)
- [ ] Training data preparation + time-series split
- [ ] Train Model A (foreign flow classifier)
- [ ] Train Model B (stock outperform)
- [ ] SHAP integration + backtesting

### Phase 4 — Output & Dashboard (1–2 minggu)
- [ ] FastAPI endpoints
- [ ] Explanation generator dengan RAG + LLM
- [ ] Streamlit dashboard
- [ ] Telegram bot alert

### Phase 5 — Production (ongoing)
- [ ] Docker containerization
- [ ] Monitoring model drift
- [ ] A/B testing model versions
- [ ] Source score calibration dari track record

---

*Last updated: Juni 2025 | Scope: LQ45 | Stack: Python*
