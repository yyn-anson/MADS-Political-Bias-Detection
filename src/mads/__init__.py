"""MADS: conditional multi-agent debate for political bias detection."""

from .engine import MADSAnalyzer
from .types import Article, BiasLabel

__all__ = ["Article", "BiasLabel", "MADSAnalyzer"]
__version__ = "2.0.0"
