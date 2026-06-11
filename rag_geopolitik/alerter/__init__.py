"""Telegram alert bot for high-confidence signals.

Spec section 9.2 — sends alerts when outperform probability > 75%
and confidence > 70%.
"""

from rag_geopolitik.alerter.bot import AlertBot

__all__ = ["AlertBot"]