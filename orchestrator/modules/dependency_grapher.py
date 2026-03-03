"""
modules/dependency_grapher.py — Module 5: Dependency Grapher
─────────────────────────────────────────────────────────────
Runs every 12 hours.

  1. Fetches all papers and their "Related Papers" relational connections from
     the Notion "Paper Tracker" database.
  2. Builds a directed graph with NetworkX.
  3. Renders an interactive HTML file with PyVis.
  4. Saves the HTML to /app/static/graph.html (served by graph-server).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from .notion_client_wrapper import NotionClientWrapper

logger = logging.getLogger(__name__)

# Where to write the output file (shared Docker volume with graph-server)
STATIC_DIR = Path(os.environ.get("GRAPH_STATIC_DIR", "/app/static"))
GRAPH_FILE = STATIC_DIR / "graph.html"


class DependencyGrapher:
    """Module 5: Builds an interactive paper dependency graph."""

    def __init__(self) -> None:
        self.notion = NotionClientWrapper()
        self.paper_tracker_db = os.environ["NOTION_PAPER_TRACKER_DB_ID"]

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Fetch data, build graph, render HTML."""
        logger.info("Dependency Grapher: fetching papers …")
        pages = self.notion.query_database(self.paper_tracker_db)
        logger.info("Dependency Grapher: %d paper(s) found.", len(pages))

        if not pages:
            logger.warning("Dependency Grapher: no pages — skipping graph generation.")
            return

        graph = self._build_graph(pages)
        self._render_html(graph)
        logger.info("Dependency Grapher: graph written to %s.", GRAPH_FILE)

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self, pages: list[dict]) -> nx.DiGraph:
        """Return a directed NetworkX graph from Notion page data."""
        graph = nx.DiGraph()

        # id → title mapping (build first so edges can reference titles)
        id_to_title: dict[str, str] = {}
        for page in pages:
            pid = page["id"]
            title = self._get_title(page["properties"])
            status = self._get_select(page["properties"], "Status")
            id_to_title[pid] = title
            graph.add_node(
                pid,
                label=title[:40],  # truncate for readability
                title=title,       # full title shown on hover
                status=status,
                color=self._status_colour(status),
            )

        # Add directed edges from "Related Papers" relation
        for page in pages:
            pid = page["id"]
            related = self._get_relation(page["properties"], "Related Papers")
            for related_id in related:
                if related_id in id_to_title:
                    graph.add_edge(pid, related_id)
                else:
                    logger.debug(
                        "Dependency Grapher: related page %s not in DB snapshot.",
                        related_id,
                    )

        return graph

    # ── HTML rendering ────────────────────────────────────────────────────────

    def _render_html(self, graph: nx.DiGraph) -> None:
        """Render the NetworkX graph to an interactive PyVis HTML file."""
        STATIC_DIR.mkdir(parents=True, exist_ok=True)

        net = Network(
            height="900px",
            width="100%",
            directed=True,
            bgcolor="#1a1a2e",
            font_color="white",
            notebook=False,
        )

        # Copy nodes with attributes
        for node_id, data in graph.nodes(data=True):
            net.add_node(
                node_id,
                label=data.get("label", node_id),
                title=data.get("title", node_id),
                color=data.get("color", "#6c63ff"),
            )

        # Copy edges
        for src, dst in graph.edges():
            net.add_edge(src, dst, color="#888888", arrows="to")

        # Physics options for a cleaner layout
        net.set_options(
            """{
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -8000,
                  "springLength": 200,
                  "springConstant": 0.04
                },
                "minVelocity": 0.75
              },
              "interaction": {
                "hover": true,
                "tooltipDelay": 100
              }
            }"""
        )

        net.save_graph(str(GRAPH_FILE))

    # ── Status colour mapping ─────────────────────────────────────────────────

    @staticmethod
    def _status_colour(status: str) -> str:
        colours = {
            "s0-inbox": "#f4a261",
            "s1-skim": "#e9c46a",
            "s2-read": "#2a9d8f",
            "s3-annotated": "#264653",
            "s4-mastered": "#6c63ff",
        }
        return colours.get(status, "#888888")

    # ── Property helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_title(props: dict) -> str:
        for value in props.values():
            if value.get("type") == "title":
                try:
                    return value["title"][0]["plain_text"]
                except (KeyError, IndexError):
                    return "Untitled"
        return "Untitled"

    @staticmethod
    def _get_select(props: dict, key: str) -> str:
        try:
            return props[key]["select"]["name"]
        except (KeyError, TypeError):
            return ""

    @staticmethod
    def _get_relation(props: dict, key: str) -> list[str]:
        try:
            return [r["id"] for r in props[key]["relation"]]
        except (KeyError, TypeError):
            return []
