"""ML prediction engine: Model A (foreign flow) and Model B (stock outperform).

Spec section 7.
"""

from rag_geopolitik.models.flow_model import FlowModel
from rag_geopolitik.models.stock_model import StockModel

__all__ = ["FlowModel", "StockModel"]