"""Quick demo: run the full pipeline with synthetic data — no external deps needed.

Usage:
    python3 demo_pipeline.py
"""

from __future__ import annotations
import sys
sys.path.insert(0, '.')

from datetime import datetime, timezone
from rag_geopolitik.collector import NewsCrawler
from rag_geopolitik.config import load_source_registry
from rag_geopolitik.constants import map_entities_to_tickers
from rag_geopolitik.extraction import EventExtractor
from rag_geopolitik.features import FeatureBuilder
from rag_geopolitik.schemas import RawArticle, ExtractedEvent, Magnitude, SentimentDirection


def main():
    print("=" * 60)
    print("RAG Geopolitik & Investasi")
    print("=" * 60)

    # 1. Show loaded sources
    registry = load_source_registry()
    print(f"\n📰 {len(registry)} news sources loaded:")
    for s in registry.values():
        print(f"   {s.id:20s} | cred: {s.credibility_score:.2f} | {s.language}")

    # 2. Build synthetic articles
    print("\n📝 Building synthetic articles...")
    source = registry["reuters"]
    articles = [
        NewsCrawler.build_article(
            source=registry["reuters"],
            title="China Announces Major Stimulus Package for Infrastructure",
            body=(
                "China's government announced a comprehensive stimulus package "
                "aimed at boosting infrastructure spending. The package includes "
                "significant investments in nickel and coal supply chains. "
                "Analysts say this will increase demand for nickel and coal, "
                "benefiting Indonesian mining companies. The stimulus is seen "
                "as positive for emerging markets, with potential foreign "
                "inflows expected. Market sentiment is bullish."
            ),
            url="https://reuters.com/article/china-stimulus",
            published_at=datetime.now(timezone.utc),
            language="en",
            category="macro",
        ),
        NewsCrawler.build_article(
            source=registry["kontan"],
            title="IHSG Menguat Didorong Sentimen Positif China",
            body=(
                "Indeks Harga Saham Gabungan (IHSG) menguat didorong oleh "
                "sentimen positif dari stimulus China. Saham-saham komoditas "
                "seperti ANTM dan ADRO mencatat kenaikan signifikan. Investor "
                "asing mulai melakukan aksi beli di pasar Indonesia. "
                "Analis memperkirakan trend positif ini akan berlanjut. "
                "Volume perdagangan meningkat 1.8x dari rata-rata 20 hari."
            ),
            url="https://kontan.co.id/ihsg-menguat",
            published_at=datetime.now(timezone.utc),
            language="id",
            category="macro",
        ),
        NewsCrawler.build_article(
            source=registry["bisnis"],
            title="Nickel Price Surges as China Demand Outlook Improves",
            body=(
                "Nickel prices on the LME surged 2.3% today as China's "
                "stimulus package boosted demand outlook for the metal. "
                "This is positive for Indonesian nickel miners ANTM, INCO, "
                "and MDKA. The rally in commodity prices supports the "
                "positive outlook for Indonesian mining stocks."
            ),
            url="https://bisnis.com/nickel-surge",
            published_at=datetime.now(timezone.utc),
            language="en",
            category="commodity",
        ),
    ]
    print(f"   Created {len(articles)} articles")

    # 3. Extract events
    print("\n🔍 Extracting events...")
    extractor = EventExtractor()
    events = extractor.extract_batch(articles)
    print(f"   Extracted {len(events)} events:")
    for e in events:
        print(f"   - [{e.event_type}] sentiment={e.sentiment_score:+.2f} "
              f"({e.sentiment_direction.value}) tickers={e.tickers_affected}")
        print(f"     snippet: {e.raw_text_snippet[:100]}...")

    # 4. Build features
    print("\n⚙️  Building feature vectors...")
    builder = FeatureBuilder()
    features = builder.build_all(events=events)
    print(f"   Generated {len(features)} features across 4 buckets:")
    print(f"   {dict(list(features.items())[:8])}")  # first 8 features

    # 5. Entity-to-ticker mapping demo
    print("\n🔗 Entity → Ticker mapping:")
    entities_sample = ["nickel", "bca", "china stimulus", "coal"]
    mapped = map_entities_to_tickers(entities_sample)
    print(f"   Entities: {entities_sample}")
    print(f"   Mapped tickers: {mapped}")

    # 6. Build per-ticker features
    print("\n📊 Per-ticker feature examples:")
    for ticker in ["ANTM", "BBCA", "ADRO"]:
        ticker_feats = builder.build_for_ticker(ticker, events=events)
        print(f"   {ticker}: {len(ticker_feats)} features")

    # 7. Summary
    print("\n" + "=" * 60)
    print("✅ Demo complete — pipeline is fully wired end-to-end!")
    print("=" * 60)
    print("\nNext steps to get real predictions:")
    print("   1. pip install -r requirements.txt")
    print("   2. Create a .env file with your API keys")
    print("   3. Run real crawlers (implement _fetch_source)")
    print("   4. Train models with historical data")
    print("   5. Start API: uvicorn run ...")
    print("\nOr explore interactively: python3 -c \"from rag_geopolitik import *\"")


if __name__ == "__main__":
    main()