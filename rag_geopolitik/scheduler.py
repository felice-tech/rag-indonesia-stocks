"""APScheduler-based periodic tasks: crawl, ingest, extract, and predict.

Spec sections 2.2, 3.2 — runs every 30 minutes / daily.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rag_geopolitik.alerter.bot import AlertBot
from rag_geopolitik.collector.crawler import NewsCrawler
from rag_geopolitik.engine.ingestor import Ingestor
from rag_geopolitik.extraction.extractor import EventExtractor
from rag_geopolitik.features.builder import FeatureBuilder
from rag_geopolitik.features.store import FeatureStore
from rag_geopolitik.models.flow_model import FlowModel
from rag_geopolitik.models.stock_model import StockModel
from rag_geopolitik.recommendation.recommender import Recommender

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:
    BackgroundScheduler = None  # type: ignore[assignment,misc]


LQ45_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "ANTM", "ADRO", "TLKM", "UNVR",
    "ICBP", "INCO", "MDKA", "AALI", "LSIP", "ITMG", "BUMI", "MEDC",
    "ELSA", "ASII", "UNTR", "GGRM", "HMSP", "KLBF", "SIDO", "CPIN",
    "JPFA", "INDF", "MYOR", "ACES", "MNCN", "SCMA", "EXCL", "ISAT",
    "TOWR", "PGAS", "PTBA",
]


class PipelineScheduler:
    """Orchestrates periodic data collection, ingestion, and prediction tasks.

    Parameters
    ----------
    crawler : NewsCrawler
    ingestor : Ingestor
    extractor : EventExtractor
    feature_builder : FeatureBuilder
    feature_store : FeatureStore
    flow_model : FlowModel
    stock_model : StockModel
    recommender : Recommender
    alert_bot : AlertBot, optional
    """

    def __init__(
        self,
        crawler: NewsCrawler,
        ingestor: Ingestor,
        extractor: EventExtractor,
        feature_builder: FeatureBuilder,
        feature_store: FeatureStore,
        flow_model: FlowModel,
        stock_model: StockModel,
        recommender: Recommender,
        alert_bot: AlertBot | None = None,
    ) -> None:
        self.crawler = crawler
        self.ingestor = ingestor
        self.extractor = extractor
        self.feature_builder = feature_builder
        self.feature_store = feature_store
        self.flow_model = flow_model
        self.stock_model = stock_model
        self.recommender = recommender
        self.alert_bot = alert_bot

        self._scheduler = (
            BackgroundScheduler() if BackgroundScheduler is not None else None
        )

    # ------------------------------------------------------------------ #
    # Job definitions
    # ------------------------------------------------------------------ #
    def crawl_and_ingest_job(self) -> None:
        """Crawl all sources, ingest into RAG, and extract events.

        Runs every 30 minutes.
        """
        print(f"[{datetime.now(timezone.utc).isoformat()}] Starting crawl...")

        # 1. Crawl
        articles = self.crawler.crawl_all()
        print(f"  Crawled {len(articles)} new articles.")

        # 2. Ingest into Qdrant
        n_chunks = self.ingestor.ingest_batch(articles)
        print(f"  Ingested {n_chunks} chunks into Qdrant.")

        # 3. Extract events
        events = self.extractor.extract_batch(articles)
        print(f"  Extracted {len(events)} events.")

    def daily_prediction_job(self) -> None:
        """Run full prediction pipeline and send alerts.

        Runs daily at market close.
        """
        print(f"[{datetime.now(timezone.utc).isoformat()}] Running daily prediction...")

        # 1. Build market-level features
        features = self.feature_builder.build_all(events=[])

        # 2. Predict foreign flow
        flow_pred = self.flow_model.predict(features)
        print(f"  Flow direction: {flow_pred.direction.value} (conf: {flow_pred.confidence:.0%})")

        # 3. Build per-ticker features and predict
        opportunities: list = []
        for ticker in LQ45_TICKERS:
            ticker_feats = self.feature_builder.build_for_ticker(ticker, events=[])
            opp = self.stock_model.predict(ticker, ticker_feats)
            opportunities.append(opp)

            # Store features
            today = datetime.now(timezone.utc).date()
            self.feature_store.put_ticker_features(ticker, today, ticker_feats)

        print(f"  Generated {len(opportunities)} stock opportunities.")

        # 4. Alert high-confidence signals
        if self.alert_bot is not None:
            sent = self.alert_bot.notify_batch(opportunities, flow_pred)
            print(f"  Sent {sent} Telegram alerts.")

    # ------------------------------------------------------------------ #
    # Start / stop
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Register jobs and start the scheduler."""
        if self._scheduler is None:
            print("APScheduler not available — skipping scheduler start.")
            return

        # Every 30 minutes
        self._scheduler.add_job(
            self.crawl_and_ingest_job,
            "interval",
            minutes=30,
            id="crawl_ingest",
            replace_existing=True,
        )

        # Daily at 16:30 WIB (UTC+7 → 09:30 UTC)
        self._scheduler.add_job(
            self.daily_prediction_job,
            "cron",
            hour=9,
            minute=30,
            id="daily_prediction",
            replace_existing=True,
        )

        self._scheduler.start()
        print("Pipeline scheduler started.")

    def stop(self) -> None:
        """Gracefully shut down the scheduler."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            print("Pipeline scheduler stopped.")