"""Telegram alert bot for high-confidence stock signals.

Spec section 9.2 — sends formatted alerts when model confidence exceeds
thresholds.
"""

from __future__ import annotations

from rag_geopolitik.config import get_settings
from rag_geopolitik.schemas import StockOpportunity, FlowPrediction

try:
    from telegram import Bot as TelegramBot
except ImportError:
    TelegramBot = None  # type: ignore[assignment,misc]


_ALERT_TEMPLATE = """
🔔 *Signal Alert — {ticker}*

📊 Outperform Prob: {prob:.0%}
🎯 Confidence: {conf:.0%}
🌊 Foreign Flow: {flow}

*Alasan:*
{bullets}

⚠️ _Bukan rekomendasi investasi. DYOR._
""".strip()


class AlertBot:
    """Sends Telegram alerts for high-confidence opportunities.

    Parameters
    ----------
    token : str, optional
        Telegram bot token. Falls back to ``settings.telegram_bot_token``.
    chat_id : str, optional
        Target chat ID. Falls back to ``settings.telegram_chat_id``.
    min_prob : float
        Minimum outperform probability to trigger alert (default 0.75).
    min_confidence : float
        Minimum confidence score to trigger alert (default 0.70).
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        min_prob: float | None = None,
        min_confidence: float | None = None,
    ) -> None:
        settings = get_settings()
        self.token = token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id
        self.min_prob = min_prob or settings.alert_min_outperform_prob
        self.min_confidence = min_confidence or settings.alert_min_confidence

        self._bot = (
            TelegramBot(token=self.token)
            if TelegramBot is not None and self.token
            else None
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def notify_opportunity(
        self,
        opp: StockOpportunity,
        flow_prediction: FlowPrediction | None = None,
    ) -> bool:
        """Send an alert for a single opportunity if thresholds are met.

        Parameters
        ----------
        opp : StockOpportunity
        flow_prediction : FlowPrediction, optional

        Returns
        -------
        bool
            ``True`` if the alert was sent.
        """
        if not self._should_alert(opp):
            return False

        bullets = "\n".join(f"- {b}" for b in opp.explanation_bullets[:5])
        flow_str = (
            flow_prediction.direction.value if flow_prediction else "N/A"
        )

        message = _ALERT_TEMPLATE.format(
            ticker=opp.ticker,
            prob=opp.outperform_probability,
            conf=opp.confidence,
            flow=flow_str,
            bullets=bullets,
        )
        return self._send(message)

    def notify_batch(
        self,
        opportunities: list[StockOpportunity],
        flow_prediction: FlowPrediction | None = None,
    ) -> int:
        """Send alerts for all qualifying opportunities; returns count sent."""
        sent = 0
        for opp in opportunities:
            if self.notify_opportunity(opp, flow_prediction):
                sent += 1
        return sent

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _should_alert(self, opp: StockOpportunity) -> bool:
        """Check if the opportunity meets alert thresholds."""
        return (
            opp.outperform_probability >= self.min_prob
            and opp.confidence >= self.min_confidence
        )

    def _send(self, text: str) -> bool:
        """Send a plain-text message to the configured chat."""
        if self._bot is None or not self.chat_id:
            return False
        try:
            self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="Markdown",
            )
            return True
        except Exception:
            return False