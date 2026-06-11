"""Domain constants: event taxonomy, entity->ticker map, and risk flags.

Mirrors spec sections 4.2, 4.3, and 11.3.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Event taxonomy (spec 4.2)
# --------------------------------------------------------------------------- #
EVENT_TYPES: dict[str, list[str]] = {
    "macro": [
        "rate_decision",  # Fed, BI rate
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
    ],
}

# Flat set of every valid event subtype for quick validation.
ALL_EVENT_SUBTYPES: set[str] = {
    subtype for subtypes in EVENT_TYPES.values() for subtype in subtypes
}


# --------------------------------------------------------------------------- #
# Entity -> ticker mapping (spec 4.3)
# --------------------------------------------------------------------------- #
ENTITY_TICKER_MAP: dict[str, list[str]] = {
    # Commodity exposure
    "nickel": ["ANTM", "INCO", "MDKA"],
    "cpo": ["AALI", "LSIP", "SIMP"],
    "coal": ["ADRO", "PTBA", "ITMG", "BUMI"],
    "gold": ["ANTM", "MDKA"],
    "crude oil": ["MEDC", "ELSA"],
    # Macro / sector exposure
    "bank indonesia": ["BBCA", "BBRI", "BMRI", "BBNI"],
    "fed rate": ["BBCA", "BBRI", "BMRI", "BBNI"],
    "china stimulus": ["ANTM", "INCO", "ADRO"],
    "usd idr": ["BBCA", "BBRI", "UNVR", "ICBP"],
    # Direct mention
    "bank central asia": ["BBCA"],
    "bca": ["BBCA"],
    "telkom": ["TLKM"],
}


def map_entities_to_tickers(entities: list[str]) -> list[str]:
    """Resolve a list of entity strings to affected LQ45 tickers (deduped)."""
    tickers: list[str] = []
    for entity in entities:
        key = entity.strip().lower()
        for ticker in ENTITY_TICKER_MAP.get(key, []):
            if ticker not in tickers:
                tickers.append(ticker)
    return tickers


# --------------------------------------------------------------------------- #
# Risk flags (spec 11.3)
# --------------------------------------------------------------------------- #
RISK_FLAGS: dict[str, str] = {
    "low_novelty": "Event kemungkinan sudah priced-in (novelty < 0.4)",
    "single_source": "Hanya 1 sumber mengkonfirmasi, perlu validasi",
    "earnings_window": "Laporan keuangan akan rilis dalam 5 hari",
    "high_vix": "VIX > 25, market sedang fear mode",
    "low_liquidity": "Volume rata-rata rendah, prediksi kurang reliable",
}
