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
      "gravitationalConstant": -35000,
      "centralGravity": 0.04,
      "springLength": 420,
      "springConstant": 0.025,
      "damping": 0.14,
      "avoidOverlap": 0.6
    },
    "minVelocity": 0.3,
    "stabilization": { "iterations": 350 }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 80,
    "navigationButtons": true,
    "keyboard": true
  },
  "edges": {
    "smooth": { "type": "continuous" },
    "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
    "font": { "size": 0 }
  }
}"""


def _notion_url(page_id: str) -> str:
    """Return the canonical Notion page URL for a page id."""
    clean = page_id.replace("-", "")
    return f"https://www.notion.so/{clean}"


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
            arxiv  = self._get_text(page["properties"], "ArXiv ID")
            authors = self._get_text(page["properties"], "Authors")
            id_to_title[pid] = title
            g.add_node(
                pid,
                label=self._truncate(title, 35),
                title=f"📄 {title}\nStatus: {status}",
                color=_PAPER_STATUS_COLOURS.get(status, "#888888"),
                shape="box",
                size=20,
                node_kind="paper",
                notion_url=_notion_url(pid),
                meta=json.dumps({
                    "kind": "paper",
                    "title": title,
                    "status": status,
                    "arxiv": arxiv,
                    "authors": authors,
                    "notion_url": _notion_url(pid),
                }),
            )

        # Add SB concept nodes
        for page in sb_pages:
            pid          = page["id"]
            title        = self._get_title(page["properties"])
            concept_type = self._get_select(page["properties"], "Type")
            hub          = self._get_text(page["properties"], "Suggested Hub")
            summary      = self._get_text_full(page["properties"], "Summary")
            body_sections = self._parse_page_blocks(pid)
            tags_raw     = page["properties"].get("Tags", {})
            tags: list[str] = []
            try:
                tags = [t["name"] for t in tags_raw.get("multi_select", [])]
            except Exception:
                pass
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
                notion_url=_notion_url(pid),
                meta=json.dumps({
                    "kind": "concept",
                    "title": title,
                    "type": concept_type,
                    "hub": hub,
                    "summary": summary,
                    "tags": tags,
                    "notion_url": _notion_url(pid),
                    "body_sections": body_sections,
                }),
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
                        notion_url="",
                        meta=json.dumps({
                            "kind": "edge",
                            "relation_type": "source",
                            "rationale": "",
                            "confidence": None,
                            "status": "structural",
                            "notion_url": "",
                        }),
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
                notion_url=_notion_url(edge_page["id"]),
                meta=json.dumps({
                    "kind": "edge",
                    "relation_type": rel_type,
                    "rationale": rationale,
                    "confidence": confidence,
                    "status": status,
                    "notion_url": _notion_url(edge_page["id"]),
                }),
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
            arxiv  = self._get_text(page["properties"], "ArXiv ID")
            authors = self._get_text(page["properties"], "Authors")
            id_to_title[pid] = title
            g.add_node(
                pid,
                label=self._truncate(title, 35),
                title=f"📄 {title}\nStatus: {status}",
                color=_PAPER_STATUS_COLOURS.get(status, "#888888"),
                shape="box",
                size=20,
                node_kind="paper",
                notion_url=_notion_url(pid),
                meta=json.dumps({
                    "kind": "paper",
                    "title": title,
                    "status": status,
                    "arxiv": arxiv,
                    "authors": authors,
                    "notion_url": _notion_url(pid),
                }),
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
            summary = self._get_text_full(props, "Summary")
            body_sections = self._parse_page_blocks(pid)
            tags_raw = props.get("Tags", {})
            tags: list[str] = []
            try:
                tags = [t["name"] for t in tags_raw.get("multi_select", [])]
            except Exception:
                pass
            g.add_node(
                pid,
                label=self._truncate(title, 40),
                title=tooltip,
                color=colour,
                shape="ellipse",
                size=15,
                node_kind="ki_concept",
                notion_url=_notion_url(pid),
                meta=json.dumps({
                    "kind": "ki_concept",
                    "title": title,
                    "type": concept_type,
                    "verification_status": v_status,
                    "graph_link_status": graph_link_status,
                    "summary": summary,
                    "tags": tags,
                    "notion_url": _notion_url(pid),
                    "body_sections": body_sections,
                }),
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
                        notion_url="",
                        meta=json.dumps({
                            "kind": "edge",
                            "relation_type": "source",
                            "rationale": "",
                            "confidence": None,
                            "status": "structural",
                            "notion_url": "",
                        }),
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
                        notion_url="",
                        meta=json.dumps({
                            "kind": "edge",
                            "relation_type": rel_type,
                            "rationale": rationale,
                            "confidence": confidence,
                            "status": "ai-proposed",
                            "notion_url": "",
                        }),
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
            height="100vh",
            width="100%",
            directed=True,
            bgcolor="#1a1a2e",
            font_color="white",
            notebook=False,
        )

        # Build metadata lookup: node_id -> parsed meta dict
        node_meta: dict[str, dict] = {}
        degree_map = dict(graph.degree())
        for node_id, data in graph.nodes(data=True):
            deg = degree_map.get(node_id, 0)
            base_size = data.get("size", 15)
            scaled_size = round(base_size + min(deg * 1.5, 22))
            net.add_node(
                node_id,
                label=data.get("label", str(node_id)),
                title=data.get("title", str(node_id)),
                color=data.get("color", "#6c63ff"),
                shape=data.get("shape", "ellipse"),
                size=scaled_size,
            )
            raw_meta = data.get("meta", "{}")
            try:
                m = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
                m["degree"] = deg
                node_meta[node_id] = m
            except Exception:
                node_meta[node_id] = {"degree": deg}

        # Build edge metadata lookup: "src||dst" -> parsed meta dict
        edge_meta: dict[str, dict] = {}
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
            raw_meta = data.get("meta", "{}")
            try:
                em = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            except Exception:
                em = {}
            em["_src"] = node_meta.get(src, {}).get("title", src)
            em["_dst"] = node_meta.get(dst, {}).get("title", dst)
            em["_src_id"] = src
            em["_dst_id"] = dst
            edge_meta[f"{src}||{dst}"] = em

        net.set_options(_PHYSICS_OPTIONS)
        net.save_graph(str(output_path))

        # ── Build the injected HTML/CSS/JS ────────────────────────────────
        node_meta_js  = json.dumps(node_meta,  ensure_ascii=False)
        edge_meta_js  = json.dumps(edge_meta,  ensure_ascii=False)

        inject = f"""
<!-- ═══════════════ INJECTED PANEL ═══════════════ -->
<script id="GRAPH_NODE_META" type="application/json">{node_meta_js}</script>
<script id="GRAPH_EDGE_META" type="application/json">{edge_meta_js}</script>

<style>
/* ── Side panel ── */
#kg-panel {{
  position: fixed;
  top: 0; right: 0;
  width: 400px;
  height: 100vh;
  background: #1e1e30;
  border-left: 1px solid #3a3a5e;
  display: flex;
  flex-direction: column;
  transform: translateX(100%);
  transition: transform 0.28s cubic-bezier(0.4,0,0.2,1);
  z-index: 10000;
  font-family: 'Segoe UI', sans-serif;
  color: #e0e0f0;
  overflow: hidden;
}}
#kg-panel.open {{
  transform: translateX(0);
}}
#kg-panel-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px 10px;
  background: #252540;
  border-bottom: 1px solid #3a3a5e;
  flex-shrink: 0;
}}
#kg-panel-header h2 {{
  margin: 0;
  font-size: 0.92rem;
  font-weight: 600;
  color: #c7c7e8;
  line-height: 1.3;
  max-width: 340px;
  word-break: break-word;
}}
#kg-panel-close {{
  background: none;
  border: none;
  color: #888;
  font-size: 1.3rem;
  cursor: pointer;
  padding: 0 4px;
  line-height: 1;
  flex-shrink: 0;
}}
#kg-panel-close:hover {{ color: #e0e0f0; }}
#kg-panel-body {{
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  font-size: 0.85rem;
  line-height: 1.6;
}}
#kg-panel-body::-webkit-scrollbar {{ width: 5px; }}
#kg-panel-body::-webkit-scrollbar-track {{ background: #1e1e30; }}
#kg-panel-body::-webkit-scrollbar-thumb {{ background: #444466; border-radius: 3px; }}
.kg-badge {{
  display: inline-block;
  padding: 2px 9px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
  margin-bottom: 12px;
  letter-spacing: 0.03em;
}}
.kg-row {{
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
  align-items: flex-start;
}}
.kg-row .kg-lbl {{
  color: #888;
  min-width: 90px;
  flex-shrink: 0;
  font-size: 0.78rem;
  padding-top: 1px;
}}
.kg-row .kg-val {{
  color: #c8c8e8;
  word-break: break-word;
}}
.kg-summary {{
  background: #252540;
  border-radius: 6px;
  padding: 10px 12px;
  margin-top: 10px;
  color: #b0b0d8;
  font-size: 0.82rem;
  line-height: 1.55;
  max-height: 220px;
  overflow-y: auto;
  white-space: pre-wrap;
}}
.kg-sections {{
  margin-top: 14px;
  border-top: 1px solid #2d2d50;
  padding-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.kg-section-label {{
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #6666aa;
  margin-bottom: 3px;
  font-weight: 600;
}}
.kg-section-body {{
  background: #252540;
  border-radius: 6px;
  padding: 8px 10px;
  color: #b0b0d8;
  font-size: 0.82rem;
  line-height: 1.55;
  white-space: pre-wrap;
  max-height: 180px;
  overflow-y: auto;
}}
.kg-tags {{
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 6px;
}}
.kg-tag {{
  background: #2d2d50;
  border: 1px solid #4a4a72;
  border-radius: 10px;
  padding: 2px 8px;
  font-size: 0.72rem;
  color: #9898c8;
}}
.kg-notion-btn {{
  display: inline-flex;
  align-items: center;
  gap: 7px;
  margin-top: 14px;
  padding: 7px 14px;
  background: #2d2d50;
  border: 1px solid #5a5a9e;
  border-radius: 7px;
  color: #b0b0e8;
  text-decoration: none;
  font-size: 0.82rem;
  cursor: pointer;
  transition: background 0.15s;
}}
.kg-notion-btn:hover {{ background: #3a3a6e; color: #fff; }}
.kg-neighbours {{
  margin-top: 14px;
  border-top: 1px solid #2d2d50;
  padding-top: 12px;
}}
.kg-neighbours h3 {{
  font-size: 0.78rem;
  color: #666;
  margin: 0 0 8px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
.kg-nb-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 0;
  border-bottom: 1px solid #252540;
  cursor: pointer;
}}
.kg-nb-item:hover .kg-nb-label {{ color: #fff; }}
.kg-nb-dot {{
  width: 9px; height: 9px;
  border-radius: 50%;
  flex-shrink: 0;
}}
.kg-nb-label {{
  font-size: 0.8rem;
  color: #aaa;
  transition: color 0.1s;
}}
.kg-nb-dir {{
  font-size: 0.7rem;
  color: #555;
  margin-left: auto;
  flex-shrink: 0;
}}
/* ── Banner ── */
#kg-banner {{
  position: fixed;
  bottom: 14px;
  right: 14px;
  background: #1e1e30cc;
  color: #666;
  padding: 4px 13px;
  border-radius: 20px;
  border: 1px solid #2d2d50;
  font-family: sans-serif;
  font-size: 0.73rem;
  z-index: 9998;
  pointer-events: none;
  backdrop-filter: blur(4px);
}}
/* ── Focus-mode overlay ── */
#kg-focus-bar {{
  position: fixed;
  bottom: 18px;
  left: 50%;
  transform: translateX(-50%);
  background: #252540;
  border: 1px solid #4a4a72;
  border-radius: 20px;
  padding: 6px 18px;
  font-family: sans-serif;
  font-size: 0.82rem;
  color: #888;
  z-index: 9999;
  display: none;
  gap: 12px;
  align-items: center;
}}
#kg-focus-bar.visible {{ display: flex; }}
#kg-focus-exit {{
  background: none;
  border: none;
  color: #9898c8;
  cursor: pointer;
  font-size: 0.82rem;
  padding: 0;
}}
#kg-focus-exit:hover {{ color: #fff; }}
/* ── Colour legend ── */
#kg-legend {{
  position: fixed;
  bottom: 50px;
  left: 14px;
  background: #1e1e30ee;
  border: 1px solid #3a3a5e;
  border-radius: 8px;
  padding: 10px 13px;
  font-family: 'Segoe UI', sans-serif;
  font-size: 0.74rem;
  color: #aaa;
  z-index: 9998;
  backdrop-filter: blur(4px);
  min-width: 136px;
}}
.kg-leg-title {{
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #555;
  margin-bottom: 7px;
  font-weight: 700;
}}
.kg-leg-group {{ display: flex; flex-direction: column; gap: 4px; }}
.kg-leg-divider {{ border-top: 1px solid #2d2d50; margin: 7px 0; }}
.kg-leg-row {{ display: flex; align-items: center; gap: 7px; color: #9898c0; }}
.kg-leg-node {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
.kg-leg-edge {{ width: 16px; height: 2px; border-radius: 1px; flex-shrink: 0; }}
/* ── Search bar ── */
#kg-search-wrap {{
  position: fixed;
  top: 14px;
  left: 14px;
  z-index: 9999;
  display: flex;
  align-items: center;
  gap: 8px;
}}
#kg-search {{
  background: #1e1e30cc;
  border: 1px solid #3a3a5e;
  border-radius: 20px;
  padding: 7px 14px;
  font-size: 0.82rem;
  color: #e0e0f0;
  outline: none;
  width: 180px;
  backdrop-filter: blur(4px);
  font-family: 'Segoe UI', sans-serif;
  transition: border-color 0.15s, width 0.2s;
}}
#kg-search::placeholder {{ color: #445; }}
#kg-search:focus {{ border-color: #6c63ff; width: 240px; }}
#kg-search-count {{
  font-size: 0.73rem;
  color: #556;
  font-family: sans-serif;
  white-space: nowrap;
}}
/* ── Loading overlay ── */
#kg-loading {{
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: #1a1a2eee;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  z-index: 20000;
  font-family: 'Segoe UI', sans-serif;
  font-size: 0.88rem;
  color: #888;
  pointer-events: none;
}}
.kg-spinner {{
  width: 22px; height: 22px;
  border: 2px solid #2d2d50;
  border-top-color: #6c63ff;
  border-radius: 50%;
  animation: kg-spin 0.7s linear infinite;
}}
@keyframes kg-spin {{ to {{ transform: rotate(360deg); }} }}
</style>

<!-- Banner -->
<div id="kg-banner">{title} — {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges</div>

<!-- Side panel -->
<div id="kg-panel">
  <div id="kg-panel-header">
    <h2 id="kg-panel-title">Detail</h2>
    <button id="kg-panel-close" title="Close">✕</button>
  </div>
  <div id="kg-panel-body" id="kg-panel-body"></div>
</div>

<!-- Focus bar -->
<div id="kg-focus-bar">
  <span id="kg-focus-label">Focus mode</span>
  <button id="kg-focus-exit">✕ Exit focus</button>
</div>

<!-- Search bar -->
<div id="kg-search-wrap">
  <input id="kg-search" type="text" placeholder="🔍 Search nodes…" autocomplete="off" />
  <span id="kg-search-count"></span>
</div>

<!-- Colour legend -->
<div id="kg-legend">
  <div class="kg-leg-title">Concepts</div>
  <div class="kg-leg-group">
    <div class="kg-leg-row"><span class="kg-leg-node" style="background:#6c63ff"></span>Theorem</div>
    <div class="kg-leg-row"><span class="kg-leg-node" style="background:#2a9d8f"></span>Definition</div>
    <div class="kg-leg-row"><span class="kg-leg-node" style="background:#f4a261"></span>Lemma</div>
    <div class="kg-leg-row"><span class="kg-leg-node" style="background:#e9c46a"></span>Algorithm</div>
    <div class="kg-leg-row"><span class="kg-leg-node" style="background:#e63946"></span>Assumption</div>
    <div class="kg-leg-row"><span class="kg-leg-node" style="background:#457b9d"></span>Proof Technique</div>
  </div>
  <div class="kg-leg-divider"></div>
  <div class="kg-leg-group">
    <div class="kg-leg-row"><span class="kg-leg-edge" style="background:#e63946"></span>depends_on</div>
    <div class="kg-leg-row"><span class="kg-leg-edge" style="background:#2a9d8f"></span>enables</div>
    <div class="kg-leg-row"><span class="kg-leg-edge" style="background:#457b9d"></span>generalizes</div>
    <div class="kg-leg-row"><span class="kg-leg-edge" style="background:#7b2d8b"></span>special_case_of</div>
    <div class="kg-leg-row"><span class="kg-leg-edge" style="background:#888888"></span>related</div>
  </div>
</div>

<!-- Loading overlay -->
<div id="kg-loading">
  <div class="kg-spinner"></div>
  <span>Stabilizing graph…</span>
</div>

<script>
(function(){{
  var NODE_META = JSON.parse(document.getElementById('GRAPH_NODE_META').textContent);
  var EDGE_META = JSON.parse(document.getElementById('GRAPH_EDGE_META').textContent);

  var panel      = document.getElementById('kg-panel');
  var panelTitle = document.getElementById('kg-panel-title');
  var panelBody  = document.getElementById('kg-panel-body');
  var focusBar   = document.getElementById('kg-focus-bar');
  var focusLabel = document.getElementById('kg-focus-label');

  document.getElementById('kg-panel-close').onclick = closePanel;
  document.getElementById('kg-focus-exit').onclick  = exitFocus;

  var currentFocusId = null;

  // ── Helpers ────────────────────────────────────────────────────────────
  function badge(text, colour) {{
    return '<span class="kg-badge" style="background:' + colour + '33;color:' + colour + ';border:1px solid ' + colour + '66">' + esc(text) + '</span>';
  }}
  function row(label, value) {{
    if (!value && value !== 0) return '';
    return '<div class="kg-row"><span class="kg-lbl">' + esc(label) + '</span><span class="kg-val">' + esc(String(value)) + '</span></div>';
  }}
  function esc(s) {{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }}
  function notionBtn(url) {{
    if (!url) return '';
    return '<a class="kg-notion-btn" href="' + url + '" target="_blank" rel="noopener">'
      + '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">'
      + '<path d="M4.459 4.208c.746.606 1.026.56 2.428.466l13.215-.793c.28 0 .047-.28-.046-.326L17.86 1.968c-.42-.326-.981-.7-2.055-.607L3.01 2.295c-.466.046-.56.28-.374.466zm.793 3.08v13.904c0 .747.373 1.027 1.214.98l14.523-.84c.841-.046.935-.56.935-1.167V6.354c0-.606-.233-.933-.748-.887l-15.177.887c-.56.047-.747.327-.747.933zm14.337.745c.093.42 0 .84-.42.888l-.7.14v10.264c-.608.327-1.168.514-1.635.514-.748 0-.935-.234-1.495-.933l-4.577-7.186v6.952L12.21 19s0 .84-1.168.84l-3.222.186c-.093-.186 0-.653.327-.746l.84-.233V9.854L7.822 9.76c-.094-.42.14-1.026.793-1.073l3.456-.233 4.764 7.279v-6.44l-1.215-.14c-.093-.514.28-.887.747-.933zM1.936 1.035l13.31-.98c1.634-.14 2.055-.047 3.082.7l4.249 2.986c.7.513.934.653.934 1.213v16.378c0 1.026-.373 1.634-1.68 1.726l-15.458.934c-.98.047-1.448-.093-1.962-.747l-3.129-4.06c-.56-.747-.793-1.306-.793-1.96V2.667c0-.839.374-1.54 1.447-1.632z"/>'
      + '</svg> Open in Notion</a>';
  }}

  // ── Node colour lookup (from vis-network nodes dataset) ───────────────
  function getNodeColour(nid) {{
    try {{
      var n = network.body.data.nodes.get(nid);
      if (!n) return '#888';
      var c = n.color;
      if (typeof c === 'string') return c;
      if (c && c.background) return c.background;
    }} catch(e) {{}}
    return '#888';
  }}

  // ── Show node panel ────────────────────────────────────────────────────
  function showNodePanel(nodeId) {{
    var m = NODE_META[nodeId];
    if (!m) return;
    panelTitle.textContent = m.title || nodeId;

    var html = '';
    var kind = m.kind || 'node';

    // Badge
    if (kind === 'paper') {{
      html += badge('Paper', '#f4a261');
      html += row('Status', m.status);
      if (m.arxiv) html += row('ArXiv', m.arxiv);
      if (m.authors) html += row('Authors', m.authors);
    }} else if (kind === 'concept') {{
      html += badge(m.type || 'Concept', '#6c63ff');
      if (m.hub) html += row('Hub', m.hub);
      if (m.tags && m.tags.length) {{
        html += '<div class="kg-row"><span class="kg-lbl">Tags</span><span class="kg-val"><div class="kg-tags">';
        m.tags.forEach(function(t){{ html += '<span class="kg-tag">' + esc(t) + '</span>'; }});
        html += '</div></span></div>';
      }}
      if (m.summary) html += '<div class="kg-summary">' + esc(m.summary) + '</div>';
      if (m.body_sections && Object.keys(m.body_sections).length) {{
        html += '<div class="kg-sections">';
        Object.keys(m.body_sections).forEach(function(sec) {{
          var txt = m.body_sections[sec];
          if (!txt) return;
          html += '<div><div class="kg-section-label">' + esc(sec) + '</div>';
          html += '<div class="kg-section-body">' + esc(txt) + '</div></div>';
        }});
        html += '</div>';
      }}
    }} else if (kind === 'ki_concept') {{
      var vcol = {{verified:'#52b788',unverified:'#f4a261',rejected:'#e63946'}}[m.verification_status] || '#888';
      html += badge(m.type || 'KI Concept', vcol);
      html += row('Verification', m.verification_status);
      html += row('Link status', m.graph_link_status);
      if (m.tags && m.tags.length) {{
        html += '<div class="kg-row"><span class="kg-lbl">Tags</span><span class="kg-val"><div class="kg-tags">';
        m.tags.forEach(function(t){{ html += '<span class="kg-tag">' + esc(t) + '</span>'; }});
        html += '</div></span></div>';
      }}
      if (m.summary) html += '<div class="kg-summary">' + esc(m.summary) + '</div>';
      if (m.body_sections && Object.keys(m.body_sections).length) {{
        html += '<div class="kg-sections">';
        Object.keys(m.body_sections).forEach(function(sec) {{
          var txt = m.body_sections[sec];
          if (!txt) return;
          html += '<div><div class="kg-section-label">' + esc(sec) + '</div>';
          html += '<div class="kg-section-body">' + esc(txt) + '</div></div>';
        }});
        html += '</div>';
      }}
    }}

    // Neighbours
    var edges = network.getConnectedEdges(nodeId);
    var nbMap = {{}}; // nid -> {{dir, edgeKind}}
    edges.forEach(function(eid) {{
      try {{
        var e = network.body.data.edges.get(eid);
        if (!e) return;
        var other = (e.from === nodeId) ? e.to : e.from;
        var dir   = (e.from === nodeId) ? '→' : '←';
        if (!nbMap[other]) nbMap[other] = [];
        nbMap[other].push(dir + (e.label ? ' ' + e.label : ''));
      }} catch(ex) {{}}
    }});

    var nbIds = Object.keys(nbMap);
    if (nbIds.length) {{
      html += '<div class="kg-neighbours"><h3>Neighbours (' + nbIds.length + ')</h3>';
      nbIds.forEach(function(nid) {{
        var nm = NODE_META[nid] || {{}};
        var col = getNodeColour(nid);
        html += '<div class="kg-nb-item" data-nid="' + esc(nid) + '">';
        html += '<div class="kg-nb-dot" style="background:' + col + '"></div>';
        html += '<span class="kg-nb-label">' + esc(nm.title || nid) + '</span>';
        html += '<span class="kg-nb-dir">' + esc(nbMap[nid].join(', ')) + '</span>';
        html += '</div>';
      }});
      html += '</div>';
    }}

    if (m.degree !== undefined) html += row('Connections', m.degree);
    html += notionBtn(m.notion_url);
    panelBody.innerHTML = html;

    // Wire neighbour click → navigate
    panelBody.querySelectorAll('.kg-nb-item').forEach(function(el) {{
      el.addEventListener('click', function() {{
        var nid = el.getAttribute('data-nid');
        network.selectNodes([nid]);
        network.focus(nid, {{scale: 1.2, animation: {{duration:400,easingFunction:'easeInOutQuad'}}}} );
        showNodePanel(nid);
        applyFocus(nid);
      }});
    }});

    openPanel(m.title || nodeId);
  }}

  // ── Show edge panel ────────────────────────────────────────────────────
  function showEdgePanel(edgeId) {{
    try {{
      var e = network.body.data.edges.get(edgeId);
      if (!e) return;
      var key = e.from + '||' + e.to;
      var m   = EDGE_META[key] || {{}};
      var srcTitle = (NODE_META[e.from] || {{}}).title || e.from;
      var dstTitle = (NODE_META[e.to]   || {{}}).title || e.to;
      var rtype = m.relation_type || e.label || 'edge';

      panelTitle.textContent = rtype;

      var ecols = {{
        depends_on:'#e63946', enables:'#2a9d8f', generalizes:'#457b9d',
        special_case_of:'#7b2d8b', related:'#888888', source:'#cccccc'
      }};
      var col = ecols[rtype] || '#888888';

      var html = '';
      html += badge(rtype, col);
      if (m.status) html += badge(m.status, m.status==='verified'?'#52b788':m.status==='ai-proposed'?'#6c63ff':'#888');
      html += '<div style="margin:10px 0 4px;">';
      html += '<div class="kg-row"><span class="kg-lbl">From</span><span class="kg-val">' + esc(srcTitle) + '</span></div>';
      html += '<div class="kg-row"><span class="kg-lbl">To</span><span class="kg-val">' + esc(dstTitle) + '</span></div>';
      if (m.confidence !== undefined && m.confidence !== null)
        html += row('Confidence', (m.confidence * 100).toFixed(0) + '%');
      if (m.rationale)
        html += '<div class="kg-summary">' + esc(m.rationale) + '</div>';
      html += '</div>';

      if (m.notion_url)
        html += notionBtn(m.notion_url);

      panelBody.innerHTML = html;
      openPanel(rtype + ': ' + srcTitle.slice(0,20) + ' → ' + dstTitle.slice(0,20));
    }} catch(ex) {{ console.error(ex); }}
  }}

  // ── Focus mode ─────────────────────────────────────────────────────────
  function applyFocus(nodeId) {{
    if (currentFocusId === nodeId) return;
    currentFocusId = nodeId;

    var connected = new Set(network.getConnectedNodes(nodeId));
    connected.add(nodeId);

    var allNodes = network.body.data.nodes.get();
    var allEdges = network.body.data.edges.get();
    var nUpdates = [], eUpdates = [];

    allNodes.forEach(function(n) {{
      var focused = connected.has(n.id);
      nUpdates.push({{
        id: n.id,
        opacity: focused ? 1 : 0.08,
        font: {{ color: focused ? '#ffffff' : '#ffffff11' }}
      }});
    }});
    allEdges.forEach(function(edge) {{
      var focused = connected.has(edge.from) && connected.has(edge.to);
      var baseCol = (edge.color && typeof edge.color === 'object') ? edge.color.color : (edge.color || '#888');
      eUpdates.push({{
        id: edge.id,
        color: focused ? baseCol : {{ color: '#ffffff08', highlight: '#ffffff08', hover: '#ffffff08' }}
      }});
    }});

    network.body.data.nodes.update(nUpdates);
    network.body.data.edges.update(eUpdates);

    var nm = NODE_META[nodeId];
    focusLabel.textContent = 'Focused: ' + (nm ? nm.title : nodeId);
    focusBar.classList.add('visible');
  }}

  function exitFocus() {{
    if (!currentFocusId) return;
    currentFocusId = null;

    var nUpdates = [], eRestores = [];
    network.body.data.nodes.get().forEach(function(n) {{
      nUpdates.push({{ id: n.id, opacity: 1, font: {{ color: '#ffffff' }} }});
    }});
    network.body.data.edges.get().forEach(function(e) {{
      eRestores.push({{ id: e.id, color: EDGE_ORIG_COLORS[e.id] || '#888888' }});
    }});
    network.body.data.nodes.update(nUpdates);
    network.body.data.edges.update(eRestores);
    focusBar.classList.remove('visible');
  }}

  // ── Panel open / close ─────────────────────────────────────────────────
  function openPanel(headerTitle) {{
    panelTitle.textContent = headerTitle || 'Detail';
    panel.classList.add('open');
  }}

  function closePanel() {{
    panel.classList.remove('open');
    exitFocus();
  }}

  // ── Wire vis-network events ─────────────────────────────────────────────
  // PyVis uses a global `network` variable — wait until it's available.
  var EDGE_ORIG_COLORS = {{}};
  function wireEvents() {{
    if (typeof network === 'undefined') {{
      setTimeout(wireEvents, 150);
      return;
    }}
    network.body.data.edges.get().forEach(function(e) {{
      var c = e.color;
      EDGE_ORIG_COLORS[e.id] = (c && typeof c === 'object') ? (c.color || '#888888') : (c || '#888888');
    }});
    network.on('click', function(params) {{
      if (params.nodes.length > 0) {{
        var nodeId = params.nodes[0];
        applyFocus(nodeId);
        showNodePanel(nodeId);
      }} else if (params.edges.length > 0) {{
        showEdgePanel(params.edges[0]);
      }} else {{
        exitFocus();
        closePanel();
      }}
    }});
    network.on('doubleClick', function(params) {{
      if (params.nodes.length > 0) {{
        var nodeId = params.nodes[0];
        network.focus(nodeId, {{ scale: 1.5, animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }} }});
      }}
    }});
    // Fit once after initial stabilisation (safe: does not run on click)
    network.on('stabilizationIterationsDone', function() {{
      network.fit({{ animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }});
      setTimeout(function() {{
        var el = document.getElementById('kg-loading');
        if (el) el.style.display = 'none';
      }}, 650);
    }});
    // Escape closes panel
    document.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape') closePanel();
    }});
    // Search / highlight
    var _si = document.getElementById('kg-search');
    var _sc = document.getElementById('kg-search-count');
    if (_si) {{
      _si.addEventListener('input', function() {{
        var q = this.value.trim().toLowerCase();
        var all = network.body.data.nodes.get();
        if (!q) {{
          network.body.data.nodes.update(all.map(function(n) {{
            return {{ id: n.id, opacity: 1, font: {{ color: '#ffffff' }} }};
          }}));
          _sc.textContent = '';
          return;
        }}
        var hits = 0;
        network.body.data.nodes.update(all.map(function(n) {{
          var meta = NODE_META[n.id] || {{}};
          var match = (meta.title || '').toLowerCase().indexOf(q) !== -1;
          if (match) hits++;
          return {{ id: n.id, opacity: match ? 1 : 0.07,
                   font: {{ color: match ? '#ffffff' : '#ffffff11' }} }};
        }}));
        _sc.textContent = hits + ' match' + (hits !== 1 ? 'es' : '');
      }});
    }}
  }}
  wireEvents();
}})();
</script>
<!-- ═══════════════ END INJECTED PANEL ═══════════════ -->
"""

        html = output_path.read_text(encoding="utf-8")
        html = html.replace("</body>", inject + "</body>", 1)
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

    def _parse_page_blocks(self, page_id: str) -> dict[str, str]:
        """Fetch page body blocks and extract labelled section text."""
        _SECTIONS = frozenset({
            "Assumptions", "Statement", "Variables",
            "Conclusion", "Source Quote", "Interpretation", "Proof Idea",
        })
        try:
            blocks = self.notion.get_block_children(page_id)
        except Exception:
            return {}
        sections: dict[str, list[str]] = {}
        current: str | None = None
        for block in blocks:
            btype = block.get("type", "")
            if btype == "heading_2":
                rt = block.get("heading_2", {}).get("rich_text", [])
                heading = "".join(seg.get("plain_text", "") for seg in rt).strip()
                if heading in _SECTIONS:
                    current = heading
                    sections.setdefault(current, [])
                else:
                    current = None
            elif btype == "paragraph" and current is not None:
                rt = block.get("paragraph", {}).get("rich_text", [])
                text = "".join(seg.get("plain_text", "") for seg in rt)
                if text.strip():
                    sections[current].append(text)
            elif btype == "equation" and current is not None:
                expr = block.get("equation", {}).get("expression", "")
                if expr.strip():
                    sections[current].append(f"$${expr}$$")
        return {k: "\n".join(v).strip() for k, v in sections.items() if v}