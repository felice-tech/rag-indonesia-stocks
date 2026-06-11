# RAG Geopolitik & Investasi

**AI-powered stock analysis for Indonesia's LQ45 market** — blends news sentiment, foreign flow analysis, and machine learning to rank stock opportunities.

## Pipeline Overview

```
📡 News RSS Feeds        📊 Yahoo Finance          🧠 ML Models
(Kontan, CNBC ID)    →   (Market + ticker    →    (XGBoost + LightGBM)
                          features)                 ↓
                                              📈 Recommendations
                                              (ranked by outperform
                                               probability + explanation)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key (FREE)
echo 'GEMINI_API_KEY="your_key_here"' > .env

# 3. Run full pipeline (train + predict + API)
python3 run_pipeline.py

# 4. Or just train + see predictions (no API)
python3 run_pipeline.py --train-only

# 5. Or start API only (if models already trained)
python3 run_pipeline.py --api-only
```

## Output Example

```
🌊 Foreign Flow: NET_SELL
   Confidence: 61%

📊 Foreign Flow Activity — today (real Yahoo Finance data):
   ──────────────────────────────────────────────────────────────────────
   🔵 TOP 5 NET BUY (by today's return):
      BBCA   return=+3.10%  shares=  543.3M  accum=IDR3164.70Bn
      KLBF   return=+2.84%  shares=   98.0M  accum=IDR71.03Bn
      TLKM   return=+2.14%  shares=  275.4M  accum=IDR790.53Bn
   ──────────────────────────────────────────────────────────────────────
   🔴 TOP 5 NET SELL (by today's return):
      MDKA   return=-7.14%  shares=   97.7M  accum=IDR241.44Bn
      INCO   return=-6.15%  shares=   19.8M  accum=IDR84.67Bn

📈 Top stock opportunities (ranked by outperform probability):
   Ticker  Prob   Conf   Why
   BBCA    90%   90%    boosted by: return_1d=+3.10, volume=0.87
   KLBF    85%   85%    boosted by: return_1d=+2.84, volume=1.17
   ...
```

## Project Architecture

| Module             | Path                             | Purpose                                                                 |
| ------------------ | -------------------------------- | ----------------------------------------------------------------------- |
| **Collector**      | `rag_geopolitik/collector/`      | RSS news crawler + deduplication                                        |
| **Engine**         | `rag_geopolitik/engine/`         | RAG pipeline: chunking, embedding (Google Gemini), Qdrant vector search |
| **Extraction**     | `rag_geopolitik/extraction/`     | NER, sentiment analysis, ticker-entity linking                          |
| **Features**       | `rag_geopolitik/features/`       | Feature engineering (4 buckets: macro, flow, ticker, news)              |
| **Models**         | `rag_geopolitik/models/`         | XGBoost (foreign flow) + LightGBM (stock outperform)                    |
| **Recommendation** | `rag_geopolitik/recommendation/` | Confidence adjuster + explanation generator (Gemini Flash Lite)         |
| **API**            | `rag_geopolitik/api/`            | FastAPI REST endpoints                                                  |
| **Dashboard**      | `rag_geopolitik/dashboard/`      | Streamlit dashboard                                                     |
| **Alerter**        | `rag_geopolitik/alerter/`        | Telegram bot alerts                                                     |

## Data Sources

| Source             | Type                                        | Credibility |
| ------------------ | ------------------------------------------- | ----------- |
| **CNBC Indonesia** | RSS (`https://www.cnbcindonesia.com/rss`)   | 0.80        |
| **Kontan**         | RSS (`https://www.kontan.co.id/rss/latest`) | 0.82        |
| **Bisnis.com**     | RSS (`https://www.bisnis.com/feed.xml`)     | 0.78        |
| **Yahoo Finance**  | Market data + ticker prices                 | —           |
| **Google News**    | Fallback aggregator                         | —           |

## ML Models

### Model A — Foreign Flow (XGBoost)

- Predicts: `NET_BUY` / `NET_SELL` / `NEUTRAL`
- Features: DXY, VIX, EM ETF flows, IHSG returns, sentiment
- Trained on 500 synthetic samples

### Model B — Stock Outperform (LightGBM)

- Predicts: probability each LQ45 stock outperforms IHSG in 5 days
- 14 features: return momentum, volume ratio, flow direction, sentiment, events
- Trained on 2000 synthetic samples

## API Endpoints

| Method | Path               | Description                 |
| ------ | ------------------ | --------------------------- |
| `GET`  | `/health`          | Health check                |
| `GET`  | `/predict/flow`    | Foreign flow prediction     |
| `POST` | `/predict/stock`   | Per-ticker stock prediction |
| `GET`  | `/recommendations` | Top ranked opportunities    |
| `GET`  | `/news/events`     | Latest extracted events     |

## Environment Variables

```env
# Required
GEMINI_API_KEY="your_google_gemini_key"

# Optional (defaults work for local dev)
QDRANT_URL="http://localhost:6333"
REDIS_URL="redis://localhost:6379/0"
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""
```

## Demo (no API keys required)

```bash
python3 demo_pipeline.py
```

Runs a quick end-to-end demo with synthetic data to verify the pipeline works.

## License

MIT
