"""Import smoke tests — guard the Store cutover wiring.

These import the rewritten pipeline packages (heavy deps are stubbed by conftest)
so a broken import / missing symbol after the Notion→Store migration fails loudly.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_ingestion_engine_imports():
    from modules.ingestion import IngestionEngine
    from modules.ingestion.concept_writer import ConceptWriter
    assert IngestionEngine is not None and ConceptWriter is not None


def test_promotion_engine_imports():
    from modules.promotion import PromotionEngine
    assert PromotionEngine is not None


def test_arxiv_sniper_imports():
    from modules.arxiv_sniper import ArXivSniper
    assert ArXivSniper is not None


def test_no_notion_modules_remain():
    import importlib
    for gone in (
        "modules.notion_client_wrapper",
        "modules.notion_parser",
        "modules.notion",
        "modules.dependency_grapher",
        "modules.promotion.edge_parser",
        "modules.ingestion.ki_writer",
    ):
        try:
            importlib.import_module(gone)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{gone} should have been deleted in the cutover")
