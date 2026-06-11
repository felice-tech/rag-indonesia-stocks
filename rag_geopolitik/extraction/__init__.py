"""Event extraction pipeline: NER, event-type classification, sentiment, entity-to-ticker linking.

Spec section 4.
"""

from rag_geopolitik.extraction.extractor import EventExtractor

__all__ = ["EventExtractor"]