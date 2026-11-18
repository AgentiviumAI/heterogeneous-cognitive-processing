"""Reusable infrastructure for PaperQA baseline experiments.

Keep package import lightweight so provider/catalog utilities can be tested without
loading the full PaperQA runtime.
"""

from .config import ModelCatalog

__all__ = ["ModelCatalog"]
