"""Start the API server with trained models.

Usage:
    python3 start_api.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_geopolitik.api import create_app
from rag_geopolitik.models.flow_model import FlowModel
from rag_geopolitik.models.stock_model import StockModel
from rag_geopolitik.recommendation import Recommender
import uvicorn

# Load trained models
flow_path = "data/models/flow_model.pkl"
stock_path = "data/models/stock_model.pkl"

flow = FlowModel(model_path=flow_path) if os.path.exists(flow_path) else FlowModel()
stock = StockModel(model_path=stock_path) if os.path.exists(stock_path) else StockModel()
rec = Recommender(stock_model=stock)

app = create_app(
    flow_model=flow,
    stock_model=stock,
    recommender=rec,
)

print("=" * 60)
print("🚀 RAG Geopolitik & Investasi API")
print(f"   Flow Model trained : {flow.is_trained}")
print(f"   Stock Model trained: {stock.is_trained}")
print("=" * 60)
print("   Server: http://localhost:6000")
print("   Docs  : http://localhost:6000/docs")
print("=" * 60)

uvicorn.run(app, host="0.0.0.0", port=6000)