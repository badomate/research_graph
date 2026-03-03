"""
modules/latex_compiler.py — Module 4: LaTeX Skeleton Compiler
──────────────────────────────────────────────────────────────
Polls the "Projects" Notion DB for rows where `Generate Skeleton` is True.
For each such project:

  1. Query all linked papers and approved theorems.
  2. Generate a structured .tex skeleton locally.
  3. Auto-populate \\begin{thebibliography} and section headers from claims.
  4. Save the .tex file to the shared WebDAV folder.
  5. Uncheck the `Generate Skeleton` property.
"""

from __future__ import annotations

import logging
import os
import textwrap
import uuid
from pathlib import Path

from webdav3.client import Client as WebDAVClient

from .notion_client_wrapper import NotionClientWrapper

logger = logging.getLogger(__name__)

# Output directory (also mounted WebDAV-synced or shared volume)
OUTPUT_DIR = Path(os.environ.get("PIPELINE_TMP_DIR", "/tmp/pipeline"))


class LaTeXCompiler:
    """Module 4: Generates a LaTeX skeleton for a research project."""

    def __init__(self) -> None:
        self.notion = NotionClientWrapper()
        self._webdav = self._build_webdav_client()
        self.projects_db = os.environ["NOTION_PROJECTS_DB_ID"]
        self.second_brain_db = os.environ["NOTION_SECOND_BRAIN_DB_ID"]
        self.paper_tracker_db = os.environ["NOTION_PAPER_TRACKER_DB_ID"]
        self.koofr_base = os.environ.get("KOOFR_PDF_PATH", "/Papers")

    # ── WebDAV ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_webdav_client() -> WebDAVClient:
        options = {
            "webdav_hostname": "https://app.koofr.net/dav/Koofr",
            "webdav_login": os.environ["KOOFR_USER"],
            "webdav_password": os.environ["KOOFR_APP_PASSWORD"],
        }
        return WebDAVClient(options)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Poll for projects with Generate Skeleton = True."""
        logger.info("LaTeX Compiler: polling for projects …")
        projects = self.notion.query_database(
            self.projects_db,
            filter={
                "property": "Generate Skeleton",
                "checkbox": {"equals": True},
            },
        )
        logger.info("LaTeX Compiler: found %d project(s) to compile.", len(projects))
        for project in projects:
            try:
                self._compile_project(project)
            except Exception:
                logger.exception("LaTeX compilation failed for %s", project["id"])

    # ── Per-project compilation ───────────────────────────────────────────────

    def _compile_project(self, project: dict) -> None:
        page_id = project["id"]
        props = project["properties"]
        project_name = self._get_title(props)
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in project_name)

        logger.info("LaTeX Compiler: compiling '%s' …", project_name)

        # 1. Fetch linked papers and theorems
        papers = self._fetch_linked_pages(props, "Papers", self.paper_tracker_db)
        theorems = self._fetch_linked_pages(props, "Theorems", self.second_brain_db)

        # 2. Generate .tex content
        tex_content = self._build_tex(project_name, papers, theorems)

        # 3. Save locally
        job_id = uuid.uuid4().hex[:8]
        local_path = OUTPUT_DIR / f"{safe_name}_{job_id}.tex"
        local_path.write_text(tex_content, encoding="utf-8")
        logger.info("LaTeX Compiler: written %s", local_path)

        # 4. Upload to WebDAV / Koofr
        remote_path = f"{self.koofr_base}/Skeletons/{local_path.name}"
        try:
            self._webdav.upload_sync(
                remote_path=remote_path,
                local_path=str(local_path),
            )
            logger.info("LaTeX Compiler: uploaded to %s", remote_path)
        except Exception:
            logger.exception("WebDAV upload failed; .tex left at %s", local_path)

        # 5. Uncheck Generate Skeleton
        self.notion.update_page(
            page_id=page_id,
            properties={"Generate Skeleton": self.notion.checkbox_prop(False)},
        )

    # ── Fetch linked pages ────────────────────────────────────────────────────

    def _fetch_linked_pages(
        self, props: dict, relation_key: str, database_id: str
    ) -> list[dict]:
        """Resolve a relation property into full page objects."""
        try:
            relations = props[relation_key]["relation"]
        except (KeyError, TypeError):
            return []

        pages = []
        for rel in relations:
            try:
                page = self.notion.get_page(rel["id"])
                pages.append(page)
            except Exception:
                logger.warning("Could not fetch page %s", rel["id"])
        return pages

    # ── .tex generation ───────────────────────────────────────────────────────

    def _build_tex(
        self, project_name: str, papers: list[dict], theorems: list[dict]
    ) -> str:
        """Return a structured LaTeX document as a string."""

        # ── Preamble ──────────────────────────────────────────────────────────
        lines: list[str] = [
            r"\documentclass[12pt,a4paper]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{amsmath, amssymb, amsthm}",
            r"\usepackage{hyperref}",
            r"\usepackage{geometry}",
            r"\geometry{margin=2.5cm}",
            "",
            r"\newtheorem{theorem}{Theorem}[section]",
            r"\newtheorem{lemma}[theorem]{Lemma}",
            r"\newtheorem{proposition}[theorem]{Proposition}",
            r"\newtheorem{corollary}[theorem]{Corollary}",
            r"\newtheorem{definition}[theorem]{Definition}",
            r"\newtheorem{remark}[theorem]{Remark}",
            "",
            rf"\title{{{self._tex_escape(project_name)} \\ \large Research Skeleton}}",
            r"\author{Auto-generated by paper\_pipeline}",
            r"\date{\today}",
            "",
            r"\begin{document}",
            r"\maketitle",
            r"\tableofcontents",
            r"\newpage",
            "",
        ]

        # ── Introduction ──────────────────────────────────────────────────────
        lines += [
            r"\section{Introduction}",
            r"% TODO: write introduction",
            "",
            r"This skeleton was auto-generated from the following sources:",
            r"\begin{itemize}",
        ]
        for p in papers:
            title = self._page_title(p)
            lines.append(rf"    \item {self._tex_escape(title)}")
        lines += [r"\end{itemize}", ""]

        # ── One section per theorem / definition ──────────────────────────────
        if theorems:
            lines += [r"\section{Key Results}", ""]
            for item in theorems:
                item_props = item.get("properties", {})
                label = self._get_title(item_props)
                content = self._get_text(item_props, "Content")
                kind = self._get_select(item_props, "Type") or "theorem"
                env = kind.lower()
                if env not in (
                    "theorem", "lemma", "proposition",
                    "corollary", "definition", "remark",
                ):
                    env = "theorem"

                lines += [
                    rf"\begin{{{env}}}[{self._tex_escape(label)}]",
                    rf"\label{{thm:{self._make_label(label)}}}",
                    content if content else r"% TODO: fill in statement",
                    rf"\end{{{env}}}",
                    "",
                ]

        # ── Bibliography ──────────────────────────────────────────────────────
        lines += [
            r"\section{Bibliography}",
            r"\begin{thebibliography}{99}",
        ]
        for i, p in enumerate(papers, start=1):
            title = self._page_title(p)
            arxiv_url = self._get_url_prop(p.get("properties", {}), "ArXiv URL")
            url_part = rf"\url{{{arxiv_url}}}" if arxiv_url else ""
            lines.append(
                rf"    \bibitem{{ref{i}}} {self._tex_escape(title)}. {url_part}"
            )
        lines += [r"\end{thebibliography}", "", r"\end{document}", ""]

        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tex_escape(text: str) -> str:
        """Escape special LaTeX characters (except already-valid math delimiters)."""
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        # Do not escape backslashes already in the content (LaTeX commands)
        result = []
        for ch in text:
            result.append(replacements.get(ch, ch))
        return "".join(result)

    @staticmethod
    def _make_label(text: str) -> str:
        """Convert text to a safe LaTeX label."""
        return "".join(c if c.isalnum() else "_" for c in text).lower()[:40]

    @staticmethod
    def _page_title(page: dict) -> str:
        props = page.get("properties", {})
        for value in props.values():
            if value.get("type") == "title":
                try:
                    return value["title"][0]["plain_text"]
                except (KeyError, IndexError):
                    pass
        return page.get("id", "Unknown")

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
    def _get_text(props: dict, key: str) -> str:
        try:
            return props[key]["rich_text"][0]["plain_text"]
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def _get_select(props: dict, key: str) -> str:
        try:
            return props[key]["select"]["name"]
        except (KeyError, TypeError):
            return ""

    @staticmethod
    def _get_url_prop(props: dict, key: str) -> str:
        try:
            return props[key]["url"] or ""
        except (KeyError, TypeError):
            return ""
