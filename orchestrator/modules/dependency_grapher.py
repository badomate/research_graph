"""
modules/dependency_grapher.py — Module 5: Dependency Grapher
─────────────────────────────────────────────────────────────
Runs every 12 hours.

Builds and renders TWO interactive HTML graphs:

  graph_verified.html  — Promoted Second Brain concepts only.
      Nodes : SB Concept pages (verified, Note Level = Concept)
              Paper Tracker pages (source papers)
      Edges : Concept → Concept  (from Edges DB, typed by Relation Type)
              Paper   → Concept  (Source Paper relation on SB pages)

  graph_inbox.html     — Knowledge Inbox concepts (unverified / in-review).
      Nodes : KI pages (all verification_status values)
              Paper Tracker pages (source papers)
      Edges : Paper → Concept  (Source Paper relation on KI pages)
              KI concept → KI concept  (Edge Suggestions JSON, AI-proposed)

Node visual encoding
────────────────────
  Papers   : rectangle shape, colour by pipeline Status
  Concepts : ellipse shape, colour by concept Type

Edge visual encoding
────────────────────
  depends_on    : red   (#e63946)
  enables       : green (#2a9d8f)
  generalizes   : blue  (#457b9d)
  special_case_of: purple (#7b2d8b)
  related       : grey  (#888888)
  source (paper→concept): dashed white (#cccccc)

Pipeline status colours (papers)
─────────────────────────────────
  s1-process-math      : #f4a261  (orange)
  s1b-waiting-attachment: #e9c46a (yellow)
  s2-extracted         : #2a9d8f  (teal)
  s2b-linked-ai        : #6c63ff  (purple)
  blocked-tags         : #e63946  (red)
  Promoted             : #52b788  (green)
  default              : #888888  (grey)

Output
──────
  /app/static/graph_verified.html
  /app/static/graph_inbox.html
  /app/static/index.html          (landing page with links to both views)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import networkx as nx
from pyvis.network import Network

from .notion_client_wrapper import NotionClientWrapper

logger = logging.getLogger(__name__)

STATIC_DIR = Path(os.environ.get("GRAPH_STATIC_DIR", "/app/static"))
GRAPH_VERIFIED_FILE = STATIC_DIR / "graph_verified.html"
GRAPH_INBOX_FILE    = STATIC_DIR / "graph_inbox.html"
INDEX_FILE          = STATIC_DIR / "index.html"

_SB_CONCEPT_LEVEL = os.environ.get("SB_CONCEPT_LEVEL", "Concept")

# ── Colour maps ───────────────────────────────────────────────────────────────

_PAPER_STATUS_COLOURS: dict[str, str] = {
    "s1-process-math":       "#f4a261",
    "s1b-waiting-attachment":"#e9c46a",
    "blocked-tags":          "#e63946",
    "s2-extracted":          "#2a9d8f",
    "s2b-linked-ai":         "#6c63ff",
    "Promoted":              "#52b788",
}

_CONCEPT_TYPE_COLOURS: dict[str, str] = {
    "Theorem":        "#6c63ff",
    "Definition":     "#2a9d8f",
    "Lemma":          "#f4a261",
    "Algorithm":      "#e9c46a",
    "Assumption":     "#e63946",
    "ProofTechnique": "#457b9d",
}

_EDGE_COLOURS: dict[str, str] = {
    "depends_on":     "#e63946",
    "enables":        "#2a9d8f",
    "generalizes":    "#457b9d",
    "special_case_of":"#7b2d8b",
    "related":        "#888888",
    "source":         "#cccccc",
}

# Shared PyVis physics options.
_PHYSICS_OPTIONS = """{
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -12000,
      "centralGravity": 0.1,
      "springLength": 220,
      "springConstant": 0.04,
      "damping": 0.09
    },
    "minVelocity": 0.5,
    "stabilization": { "iterations": 200 }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 80,
    "navigationButtons": true,
    "keyboard": true
  },
  "edges": {
    "smooth": { "type": "dynamic" },
    "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } }
  }
}"""


class DependencyGrapher:
    """
    Module 5: Builds two interactive concept/paper graphs and an index page.

    Graph 1 (graph_verified.html): verified SB concepts + Edges DB relations.
    Graph 2 (graph_inbox.html):    KI concepts + AI-proposed edge suggestions.
    """

    def __init__(self) -> None:
        self.notion              = NotionClientWrapper()
        self.paper_tracker_db    = os.environ["NOTION_PAPER_TRACKER_DB_ID"]
        self.knowledge_inbox_db  = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]
        self.second_brain_db     = os.environ["NOTION_SECOND_BRAIN_DB_ID"]
        self.edges_db            = os.environ.get("NOTION_EDGES_DB_ID", "")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Fetch all data, build both graphs, render HTML."""
        logger.info("DependencyGrapher: fetching data from Notion ...")

        papers   = self._fetch_papers()
        sb_pages = self._fetch_sb_concepts()
        ki_pages = self._fetch_ki_concepts()
        edges    = self._fetch_edges() if self.edges_db else []

        logger.info(
            "DependencyGrapher: %d paper(s), %d SB concept(s), "
            "%d KI concept(s), %d edge(s).",
            len(papers), len(sb_pages), len(ki_pages), len(edges),
        )

        STATIC_DIR.mkdir(parents=True, exist_ok=True)

        # ── Graph 1: verified SB concepts ────────────────────────────────────
        g_verified = self._build_verified_graph(papers, sb_pages, edges)
        self._render_html(
            g_verified,
            GRAPH_VERIFIED_FILE,
            title="Knowledge Graph — Verified Concepts",
        )
        logger.info("DependencyGrapher: verified graph → %s", GRAPH_VERIFIED_FILE)

        # ── Graph 2: KI inbox concepts ───────────────────────────────────────
        g_inbox = self._build_inbox_graph(papers, ki_pages)
        self._render_html(
            g_inbox,
            GRAPH_INBOX_FILE,
            title="Knowledge Graph — Inbox (Unverified)",
        )
        logger.info("DependencyGrapher: inbox graph → %s", GRAPH_INBOX_FILE)

        # ── Index page ────────────────────────────────────────────────────────
        self._render_index()
        logger.info("DependencyGrapher: index page → %s", INDEX_FILE)

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_papers(self) -> list[dict]:
        return self.notion.query_database(self.paper_tracker_db)

    def _fetch_sb_concepts(self) -> list[dict]:
        return self.notion.query_database(
            self.second_brain_db,
            filter={"property": "Note Level", "select": {"equals": _SB_CONCEPT_LEVEL}},
        )

    def _fetch_ki_concepts(self) -> list[dict]:
        return self.notion.query_database(self.knowledge_inbox_db)

    def _fetch_edges(self) -> list[dict]:
        return self.notion.query_database(self.edges_db)

    # ── Graph 1: verified ─────────────────────────────────────────────────────

    def _build_verified_graph(
        self,
        papers:   list[dict],
        sb_pages: list[dict],
        edges:    list[dict],
    ) -> nx.DiGraph:
        g: nx.DiGraph = nx.DiGraph()

        # id → title for edge resolution
        id_to_title: dict[str, str] = {}

        # Add paper nodes
        for page in papers:
            pid    = page["id"]
            title  = self._get_title(page["properties"])
            status = self._get_status(page["properties"])
            id_to_title[pid] = title
            g.add_node(
                pid,
                label=self._truncate(title, 35),
                title=f"📄 {title}\nStatus: {status}",
                color=_PAPER_STATUS_COLOURS.get(status, "#888888"),
                shape="box",
                size=20,
                node_kind="paper",
            )

        # Add SB concept nodes
        for page in sb_pages:
            pid          = page["id"]
            title        = self._get_title(page["properties"])
            concept_type = self._get_select(page["properties"], "Type")
            hub          = self._get_text(page["properties"], "Suggested Hub")
            id_to_title[pid] = title
            tooltip = f"{'⬡'} {title}\nType: {concept_type}"
            if hub:
                tooltip += f"\nHub: {hub}"
            g.add_node(
                pid,
                label=self._truncate(title, 40),
                title=tooltip,
                color=_CONCEPT_TYPE_COLOURS.get(concept_type, "#6c63ff"),
                shape="ellipse",
                size=15,
                node_kind="concept",
            )

        # Paper → Concept edges (source relation on SB pages)
        for page in sb_pages:
            concept_id  = page["id"]
            source_ids  = self._get_relation(page["properties"], "Sources")
            for paper_id in source_ids:
                if paper_id in g and concept_id in g:
                    g.add_edge(
                        paper_id,
                        concept_id,
                        color=_EDGE_COLOURS["source"],
                        dashes=True,
                        title="source",
                        edge_kind="source",
                        width=1,
                    )

        # Concept → Concept edges from Edges DB
        for edge_page in edges:
            props       = edge_page["properties"]
            from_ids    = self._get_relation(props, "From Concept")
            to_ids      = self._get_relation(props, "To Concept")
            rel_type    = self._get_select(props, "Relation Type") or "related"
            rationale   = self._get_text(props, "Rationale")
            confidence  = self._get_number(props, "AI Confidence")
            status      = self._get_select(props, "Status")

            if not from_ids or not to_ids:
                continue
            from_id = from_ids[0]
            to_id   = to_ids[0]
            if from_id not in g or to_id not in g:
                continue

            tooltip = f"{rel_type}"
            if rationale:
                tooltip += f"\n{rationale[:120]}"
            if confidence is not None:
                tooltip += f"\nConfidence: {confidence:.2f}"
            if status:
                tooltip += f"\nStatus: {status}"

            g.add_edge(
                from_id,
                to_id,
                color=_EDGE_COLOURS.get(rel_type, "#888888"),
                title=tooltip,
                label=rel_type,
                edge_kind=rel_type,
                dashes=False,
                width=2 if status == "verified" else 1,
            )

        return g

    # ── Graph 2: inbox ────────────────────────────────────────────────────────

    def _build_inbox_graph(
        self,
        papers:   list[dict],
        ki_pages: list[dict],
    ) -> nx.DiGraph:
        g: nx.DiGraph = nx.DiGraph()

        id_to_title: dict[str, str] = {}

        # Add paper nodes
        for page in papers:
            pid    = page["id"]
            title  = self._get_title(page["properties"])
            status = self._get_status(page["properties"])
            id_to_title[pid] = title
            g.add_node(
                pid,
                label=self._truncate(title, 35),
                title=f"📄 {title}\nStatus: {status}",
                color=_PAPER_STATUS_COLOURS.get(status, "#888888"),
                shape="box",
                size=20,
                node_kind="paper",
            )

        # Add KI concept nodes
        for page in ki_pages:
            pid               = page["id"]
            props             = page["properties"]
            title             = self._get_title(props)
            concept_type      = self._get_select(props, "Type")
            v_status          = self._get_select(props, "verification_status")
            graph_link_status = self._get_select(props, "graph_link_status")
            id_to_title[pid]  = title

            # Colour by verification status
            colour = {
                "verified":   "#52b788",
                "unverified": "#f4a261",
                "rejected":   "#e63946",
            }.get(v_status, "#888888")

            tooltip = (
                f"{'○'} {title}\n"
                f"Type: {concept_type}\n"
                f"Verification: {v_status}\n"
                f"Link status: {graph_link_status}"
            )
            g.add_node(
                pid,
                label=self._truncate(title, 40),
                title=tooltip,
                color=colour,
                shape="ellipse",
                size=15,
                node_kind="ki_concept",
            )

        # Paper → KI concept edges
        for page in ki_pages:
            concept_id = page["id"]
            source_ids = self._get_relation(page["properties"], "Source Paper")
            for paper_id in source_ids:
                if paper_id in g and concept_id in g:
                    g.add_edge(
                        paper_id,
                        concept_id,
                        color=_EDGE_COLOURS["source"],
                        dashes=True,
                        title="source",
                        edge_kind="source",
                        width=1,
                    )

        # KI → KI concept edges from Edge Suggestions JSON (AI-proposed)
        for page in ki_pages:
            from_id   = page["id"]
            props     = page["properties"]
            raw       = self._get_text_full(props, "Edge Suggestions")
            if not raw:
                continue

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, dict):
                continue

            for rel_type, targets in parsed.items():
                if not isinstance(targets, list):
                    continue
                for entry in targets:
                    if not isinstance(entry, dict):
                        continue
                    target_title = entry.get("target_title", "").strip()
                    rationale    = entry.get("rationale", "")
                    confidence   = entry.get("confidence", 0.0)

                    # Resolve target by title
                    to_id = next(
                        (pid for pid, t in id_to_title.items() if t == target_title),
                        None,
                    )
                    if to_id is None or to_id not in g:
                        continue

                    tooltip = f"{rel_type} (AI)\nConf: {confidence:.2f}"
                    if rationale:
                        tooltip += f"\n{rationale[:120]}"

                    g.add_edge(
                        from_id,
                        to_id,
                        color=_EDGE_COLOURS.get(rel_type, "#888888"),
                        title=tooltip,
                        label=rel_type,
                        edge_kind=rel_type,
                        dashes=True,   # dashed = AI-proposed, not human-verified
                        width=1,
                    )

        return g

    # ── HTML rendering ────────────────────────────────────────────────────────

    def _render_html(
        self,
        graph: nx.DiGraph,
        output_path: Path,
        title: str = "Knowledge Graph",
    ) -> None:
        """Render a NetworkX graph to an interactive PyVis HTML file."""
        net = Network(
            height="960px",
            width="100%",
            directed=True,
            bgcolor="#1a1a2e",
            font_color="white",
            notebook=False,
        )

        for node_id, data in graph.nodes(data=True):
            net.add_node(
                node_id,
                label=data.get("label", str(node_id)),
                title=data.get("title", str(node_id)),
                color=data.get("color", "#6c63ff"),
                shape=data.get("shape", "ellipse"),
                size=data.get("size", 15),
            )

        for src, dst, data in graph.edges(data=True):
            net.add_edge(
                src,
                dst,
                color=data.get("color", "#888888"),
                title=data.get("title", ""),
                label=data.get("label", ""),
                dashes=data.get("dashes", False),
                width=data.get("width", 1),
            )

        net.set_options(_PHYSICS_OPTIONS)

        # Inject a title banner into the HTML.
        net.save_graph(str(output_path))
        html = output_path.read_text(encoding="utf-8")
        banner = (
            f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
            f'background:#2d2d44;color:white;padding:8px 20px;border-radius:8px;'
            f'font-family:sans-serif;font-size:14px;z-index:9999;">'
            f'{title} — {graph.number_of_nodes()} nodes, '
            f'{graph.number_of_edges()} edges</div>'
        )
        html = html.replace("<body>", f"<body>\n{banner}", 1)
        output_path.write_text(html, encoding="utf-8")

    # ── Index page ────────────────────────────────────────────────────────────

    def _render_index(self) -> None:
        """Render a minimal landing page linking to both graph views."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Knowledge Graph — paper_pipeline</title>
  <style>
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #1a1a2e;
      color: #eee;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      margin: 0;
      gap: 24px;
    }
    h1 { font-size: 1.8rem; margin-bottom: 8px; }
    p  { color: #aaa; margin: 0; }
    .cards {
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
      justify-content: center;
    }
    .card {
      background: #2d2d44;
      border-radius: 12px;
      padding: 28px 36px;
      text-align: center;
      text-decoration: none;
      color: #eee;
      transition: background 0.2s, transform 0.2s;
      min-width: 200px;
    }
    .card:hover { background: #3d3d5e; transform: translateY(-4px); }
    .card .icon { font-size: 2.4rem; margin-bottom: 10px; }
    .card .label { font-size: 1.1rem; font-weight: 600; }
    .card .sub   { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .legend {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      justify-content: center;
      font-size: 0.8rem;
      color: #aaa;
    }
    .dot {
      display: inline-block;
      width: 10px; height: 10px;
      border-radius: 50%;
      margin-right: 4px;
      vertical-align: middle;
    }
  </style>
</head>
<body>
  <h1>🧠 Knowledge Graph</h1>
  <p>paper_pipeline — interactive concept visualisation</p>

  <div class="cards">
    <a class="card" href="graph_verified.html">
      <div class="icon">✅</div>
      <div class="label">Verified Concepts</div>
      <div class="sub">Second Brain · Edges DB</div>
    </a>
    <a class="card" href="graph_inbox.html">
      <div class="icon">📥</div>
      <div class="label">Inbox Concepts</div>
      <div class="sub">Knowledge Inbox · AI-proposed edges</div>
    </a>
  </div>

  <div class="legend">
    <span><span class="dot" style="background:#e63946"></span>depends_on</span>
    <span><span class="dot" style="background:#2a9d8f"></span>enables</span>
    <span><span class="dot" style="background:#457b9d"></span>generalizes</span>
    <span><span class="dot" style="background:#7b2d8b"></span>special_case_of</span>
    <span><span class="dot" style="background:#888888"></span>related</span>
    <span><span class="dot" style="background:#cccccc"></span>source (paper→concept)</span>
  </div>
</body>
</html>"""
        INDEX_FILE.write_text(html, encoding="utf-8")

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
    def _get_status(props: dict) -> str:
        """Read a status-typed property (falls back to select)."""
        for key in ("Status", "status"):
            prop = props.get(key, {})
            ptype = prop.get("type", "")
            if ptype == "status":
                try:
                    return prop["status"]["name"]
                except (KeyError, TypeError):
                    pass
            elif ptype == "select":
                try:
                    return prop["select"]["name"]
                except (KeyError, TypeError):
                    pass
        return ""

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

    @staticmethod
    def _get_text(props: dict, key: str) -> str:
        """Read first rich_text segment."""
        try:
            return props[key]["rich_text"][0]["plain_text"]
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _get_text_full(props: dict, key: str) -> str:
        """Concatenate ALL rich_text segments (handles >2000-char values)."""
        try:
            segs = props[key]["rich_text"]
            return "".join(s.get("plain_text", "") for s in segs)
        except (KeyError, TypeError):
            return ""

    @staticmethod
    def _get_number(props: dict, key: str) -> float | None:
        try:
            return float(props[key]["number"])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        return text if len(text) <= max_len else text[:max_len - 1] + "…"