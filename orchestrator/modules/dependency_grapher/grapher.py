"""
modules/dependency_grapher/grapher.py — Module 5: Dependency Grapher
─────────────────────────────────────────────────────────────────────
Builds and renders TWO interactive HTML graphs:

  graph_verified.html  — Promoted Second Brain concepts only.
  graph_inbox.html     — Knowledge Inbox concepts (unverified / in-review).

See dependency_grapher/__init__.py for full documentation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import networkx as nx
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pyvis.network import Network

from ..notion_client_wrapper import NotionClientWrapper

logger = logging.getLogger(__name__)

STATIC_DIR          = Path(os.environ.get("GRAPH_STATIC_DIR", "/app/static"))
GRAPH_VERIFIED_FILE = STATIC_DIR / "graph_verified.html"
GRAPH_INBOX_FILE    = STATIC_DIR / "graph_inbox.html"
INDEX_FILE          = STATIC_DIR / "index.html"

_SB_CONCEPT_LEVEL = os.environ.get("SB_CONCEPT_LEVEL", "Concept")

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "j2"]),
)

# ── Colour maps ───────────────────────────────────────────────────────────────

# Keys must match the Paper Tracker Status state machine (see CLAUDE.md).
_PAPER_STATUS_COLOURS: dict[str, str] = {
    "s0-inbox":               "#8d99ae",
    "s1-skim":                "#f4a261",
    "s1-processing":          "#e9c46a",
    "s1b-waiting-attachment": "#d4a373",
    "blocked-extraction":     "#e63946",
    "s2-extracted":           "#2a9d8f",
    "s2-reextract":           "#6c63ff",
    "s2-read":                "#457b9d",
    "s3-distilled":           "#52b788",
}

# Keys must match ALLOWED_CONCEPT_TYPES in extraction_schema.py.
_CONCEPT_TYPE_COLOURS: dict[str, str] = {
    "Theorem":        "#6c63ff",
    "Definition":     "#2a9d8f",
    "Lemma":          "#f4a261",
    "Algorithm":      "#e9c46a",
    "Assumption":     "#e63946",
    "Proof":          "#9d4edd",
    "ProofTechnique": "#457b9d",
}

_EDGE_COLOURS: dict[str, str] = {
    "depends_on":      "#e63946",
    "enables":         "#2a9d8f",
    "generalizes":     "#457b9d",
    "special_case_of": "#7b2d8b",
    "related":         "#888888",
    "source":          "#cccccc",
}

_PHYSICS_OPTIONS = """{
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -35000,
      "centralGravity": 0.04,
      "springLength": 420,
      "springConstant": 0.025,
      "damping": 0.14,
      "avoidOverlap": 0.6
    },
    "minVelocity": 0.75,
    "stabilization": {
      "enabled": true,
      "iterations": 350,
      "updateInterval": 25,
      "onlyDynamicEdges": false,
      "fit": true
    }
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
    clean = page_id.replace("-", "")
    return f"https://www.notion.so/{clean}"


class DependencyGrapher:
    """
    Module 5: Builds two interactive concept/paper graphs and an index page.

    Graph 1 (graph_verified.html): verified SB concepts + Edges DB relations.
    Graph 2 (graph_inbox.html):    KI concepts + AI-proposed edge suggestions.
    """

    def __init__(self) -> None:
        self.notion             = NotionClientWrapper()
        self.paper_tracker_db   = os.environ["NOTION_PAPER_TRACKER_DB_ID"]
        self.knowledge_inbox_db = os.environ["NOTION_KNOWLEDGE_INBOX_DB_ID"]
        self.second_brain_db    = os.environ["NOTION_SECOND_BRAIN_DB_ID"]
        self.edges_db           = os.environ.get("NOTION_EDGES_DB_ID", "")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
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

        g_verified = self._build_verified_graph(papers, sb_pages, edges)
        self._render_html(
            g_verified,
            GRAPH_VERIFIED_FILE,
            title="Knowledge Graph — Verified Concepts",
        )
        logger.info("DependencyGrapher: verified graph → %s", GRAPH_VERIFIED_FILE)

        g_inbox = self._build_inbox_graph(papers, ki_pages)
        self._render_html(
            g_inbox,
            GRAPH_INBOX_FILE,
            title="Knowledge Graph — Inbox (Unverified)",
        )
        logger.info("DependencyGrapher: inbox graph → %s", GRAPH_INBOX_FILE)

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
        id_to_title: dict[str, str] = {}

        for page in papers:
            pid     = page["id"]
            title   = self._get_title(page["properties"])
            status  = self._get_status(page["properties"])
            arxiv   = self._get_text(page["properties"], "ArXiv ID")
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
            tooltip = f"⬡ {title}\nType: {concept_type}"
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

        for page in sb_pages:
            concept_id = page["id"]
            source_ids = self._get_relation(page["properties"], "Sources")
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

        for edge_page in edges:
            props      = edge_page["properties"]
            from_ids   = self._get_relation(props, "From Concept")
            to_ids     = self._get_relation(props, "To Concept")
            rel_type   = self._get_select(props, "Relation Type") or "related"
            rationale  = self._get_text(props, "Rationale")
            confidence = self._get_number(props, "AI Confidence")
            status     = self._get_select(props, "Status")

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

        for page in papers:
            pid     = page["id"]
            title   = self._get_title(page["properties"])
            status  = self._get_status(page["properties"])
            arxiv   = self._get_text(page["properties"], "ArXiv ID")
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

        for page in ki_pages:
            pid               = page["id"]
            props             = page["properties"]
            title             = self._get_title(props)
            concept_type      = self._get_select(props, "Type")
            v_status          = self._get_select(props, "verification_status")
            graph_link_status = self._get_select(props, "graph_link_status")
            id_to_title[pid]  = title

            colour = {
                "verified":   "#52b788",
                "unverified": "#f4a261",
                "rejected":   "#e63946",
            }.get(v_status, "#888888")

            tooltip = (
                f"○ {title}\n"
                f"Type: {concept_type}\n"
                f"Verification: {v_status}\n"
                f"Link status: {graph_link_status}"
            )
            summary       = self._get_text_full(props, "Summary")
            body_sections = self._parse_page_blocks(pid)
            tags_raw      = props.get("Tags", {})
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

        for page in ki_pages:
            from_id = page["id"]
            props   = page["properties"]
            raw     = self._get_text_full(props, "Edge Suggestions")
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
                        dashes=True,
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
        net = Network(
            height="100vh",
            width="100%",
            directed=True,
            bgcolor="#1a1a2e",
            font_color="white",
            notebook=False,
        )

        node_meta: dict[str, dict] = {}
        degree_map = dict(graph.degree())
        for node_id, data in graph.nodes(data=True):
            deg         = degree_map.get(node_id, 0)
            base_size   = data.get("size", 15)
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
            em["_src"]    = node_meta.get(src, {}).get("title", src)
            em["_dst"]    = node_meta.get(dst, {}).get("title", dst)
            em["_src_id"] = src
            em["_dst_id"] = dst
            edge_meta[f"{src}||{dst}"] = em

        net.set_options(_PHYSICS_OPTIONS)
        net.save_graph(str(output_path))

        inject = _jinja_env.get_template("inject.html.j2").render(
            node_meta_js=json.dumps(node_meta, ensure_ascii=False),
            edge_meta_js=json.dumps(edge_meta, ensure_ascii=False),
            title=title,
            node_count=graph.number_of_nodes(),
            edge_count=graph.number_of_edges(),
        )
        html = output_path.read_text(encoding="utf-8")
        html = html.replace("</body>", inject + "</body>", 1)
        output_path.write_text(html, encoding="utf-8")

    # ── Index page ────────────────────────────────────────────────────────────

    def _render_index(self) -> None:
        html = _jinja_env.get_template("index.html.j2").render()
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
        for key in ("Status", "status"):
            prop  = props.get(key, {})
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
        try:
            return props[key]["rich_text"][0]["plain_text"]
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _get_text_full(props: dict, key: str) -> str:
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
                rt      = block.get("heading_2", {}).get("rich_text", [])
                heading = "".join(seg.get("plain_text", "") for seg in rt).strip()
                if heading in _SECTIONS:
                    current = heading
                    sections.setdefault(current, [])
                else:
                    current = None
            elif btype == "paragraph" and current is not None:
                rt   = block.get("paragraph", {}).get("rich_text", [])
                text = "".join(seg.get("plain_text", "") for seg in rt)
                if text.strip():
                    sections[current].append(text)
            elif btype == "equation" and current is not None:
                expr = block.get("equation", {}).get("expression", "")
                if expr.strip():
                    sections[current].append(f"$${expr}$$")
        return {k: "\n".join(v).strip() for k, v in sections.items() if v}
