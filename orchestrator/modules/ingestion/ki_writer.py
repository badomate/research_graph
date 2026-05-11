"""
modules/ingestion/ki_writer.py — Knowledge Inbox page creation and edge rendering.

Creates KI pages, writes edge data, renders cross-paper edge blocks,
and patches the paper page body.
"""
from __future__ import annotations

import json
import logging

from ..extraction_schema import (
    ConceptLinkResult,
    CrossPaperLinkResult,
    EdgeProposal,
    MathObject,
)
from ..notion_client_wrapper import NotionClientWrapper
from ..notion_parser import paragraph_blocks_from_latex, sanitize_statement_latex
from ..notion.block_builders import (
    append_in_batches,
    divider_block,
    heading_block,
    paragraph_blocks,
    todo_block,
)

logger = logging.getLogger(__name__)

NOTION_BLOCK_MAX_CHARS = 1900


class KnowledgeInboxWriter:
    """Writes concepts and edge data to the Knowledge Inbox Notion database."""

    def __init__(self, notion: NotionClientWrapper, knowledge_inbox_db: str) -> None:
        self.notion = notion
        self.knowledge_inbox_db = knowledge_inbox_db
        self._ki_schema: dict[str, str] | None = None

    def _get_ki_schema(self) -> dict[str, str]:
        if self._ki_schema is None:
            db = self.notion.get_database(self.knowledge_inbox_db)
            self._ki_schema = {k: v["type"] for k, v in db.get("properties", {}).items()}
            logger.debug("KI DB schema: %s", self._ki_schema)
        return self._ki_schema

    def _ki_prop(self, key: str, value: str) -> dict:
        prop_type = self._get_ki_schema().get(key, "select")
        if prop_type == "status":
            return self.notion.status_prop(value)
        return self.notion.select_prop(value)

    def create_knowledge_item(
        self,
        paper_page_id: str,
        concept: MathObject,
        hubs: dict[str, str],
        flag_reasons: list[str] | None = None,
    ) -> str:
        kind = concept.type
        title = concept.title
        source_pages_str = (
            ", ".join(str(p) for p in concept.source_pages) if concept.source_pages else ""
        )
        title_key = self.notion.get_title_property_name(self.knowledge_inbox_db)
        properties: dict = {
            title_key: self.notion.title_prop(f"{title}"),
            "Type": self.notion.select_prop(kind),
            "Status": self._ki_prop("Status", "Inbox"),
            "verification_status": self._ki_prop("verification_status", "unverified"),
            "Graph Link Status": self._ki_prop("Graph Link Status", "unlinked"),
            "Source Paper": self.notion.relation_prop([paper_page_id]),
        }
        if source_pages_str:
            properties["Source Pages"] = {"rich_text": self.notion.rich_text(source_pages_str)}
        if concept.suggested_hub:
            properties["Suggested Hub"] = {"rich_text": self.notion.rich_text(concept.suggested_hub)}
        properties["AI Confidence"] = {"number": concept.confidence}
        if concept.canonical_keywords:
            properties["Keywords"] = self.notion.multi_select_prop(concept.canonical_keywords)
        if concept.prereq_keywords:
            properties["Prereq Keywords"] = self.notion.multi_select_prop(concept.prereq_keywords)
        if concept.downstream_keywords:
            properties["Downstream Keywords"] = self.notion.multi_select_prop(concept.downstream_keywords)
        if concept.source_anchors:
            properties["Source Anchors"] = {"rich_text": self.notion.rich_text(concept.source_anchors)}
        if concept.interpretation:
            properties["Interpretation"] = {"rich_text": self.notion.rich_text(concept.interpretation)}
        if concept.proof_idea:
            properties["Proof Idea"] = {"rich_text": self.notion.rich_text(concept.proof_idea)}
        if concept.aliases:
            properties["Aliases"] = {"rich_text": self.notion.rich_text(concept.aliases)}
        if concept.assumptions:
            properties["Assumptions"] = {"rich_text": self.notion.rich_text(concept.assumptions[:2000])}
        if concept.statement_latex:
            properties["Statement LaTeX"] = {"rich_text": self.notion.rich_text(concept.statement_latex[:2000])}
        if concept.source_quotes:
            properties["Source Quote"] = {"rich_text": self.notion.rich_text(concept.source_quotes)}
        if concept.named_tools:
            properties["Named Tools"] = self.notion.multi_select_prop(concept.named_tools)
        if concept.setting:
            properties["Setting"] = self.notion.multi_select_prop(concept.setting)
        if concept.result_category:
            properties["Result Category"] = self.notion.select_prop(concept.result_category)

        new_page = self.notion.create_page(
            parent={"database_id": self.knowledge_inbox_db},
            properties=properties,
        )
        new_page_id: str = new_page["id"]
        logger.info("Created Knowledge Inbox page %s for concept '%s'.", new_page_id, title)

        body_blocks: list[dict] = []
        if flag_reasons:
            reasons_text = "\n".join(f"• {r}" for r in flag_reasons)
            callout_text = (
                f"⚠️ Quality concerns detected:\n{reasons_text}\n\n"
                "If fields are missing, add to Re-extract Hints and set s2-reextract."
            )
            body_blocks.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": callout_text[:2000]}}],
                    "icon": {"type": "emoji", "emoji": "⚠️"},
                    "color": "yellow_background",
                },
            })
        body_blocks.extend(self._review_checklist_blocks())
        body_blocks.append(heading_block("Assumptions"))
        body_blocks.extend(paragraph_blocks_from_latex(concept.assumptions))
        body_blocks.append(heading_block("Statement"))
        body_blocks.extend(paragraph_blocks_from_latex(sanitize_statement_latex(concept.statement_latex)))
        if concept.variables:
            body_blocks.append(heading_block("Variables"))
            body_blocks.extend(paragraph_blocks_from_latex(concept.variables))
        if concept.conclusion:
            body_blocks.append(heading_block("Conclusion"))
            body_blocks.extend(paragraph_blocks_from_latex(concept.conclusion))
        if concept.source_quotes:
            body_blocks.append(heading_block("Source Quote"))
            body_blocks.extend(paragraph_blocks_from_latex(concept.source_quotes))
        if concept.interpretation:
            body_blocks.append(heading_block("Interpretation"))
            body_blocks.extend(paragraph_blocks_from_latex(concept.interpretation))
        if concept.proof_idea:
            body_blocks.append(heading_block("Proof Idea"))
            body_blocks.extend(paragraph_blocks_from_latex(concept.proof_idea))

        append_in_batches(self.notion, new_page_id, body_blocks)
        return new_page_id

    def update_knowledge_item_graph_data(
        self,
        ki_page_id: str,
        link_result: ConceptLinkResult | CrossPaperLinkResult,
    ) -> None:
        if isinstance(link_result, CrossPaperLinkResult):
            self._update_ki_cross_paper(ki_page_id, link_result)
        else:
            self._update_ki_legacy(ki_page_id, link_result)

    def _update_ki_legacy(self, ki_page_id: str, link_result: ConceptLinkResult) -> None:
        edge_dict = link_result.model_dump(exclude_none=True)
        edge_dict = {k: v for k, v in edge_dict.items() if v}
        if not edge_dict:
            logger.debug("KI page %s: no edges produced — remaining 'unlinked'.", ki_page_id)
            return
        payload = edge_dict
        s = json.dumps(payload, ensure_ascii=False)
        if len(s) > NOTION_BLOCK_MAX_CHARS:
            for rel in ["related", "enables", "depends_on", "generalizes", "special_case_of"]:
                while payload.get(rel) and len(json.dumps(payload, ensure_ascii=False)) > NOTION_BLOCK_MAX_CHARS:
                    payload[rel].pop()
        edge_json = json.dumps(payload, ensure_ascii=False)
        self.notion.update_page(
            page_id=ki_page_id,
            properties={
                "Edge Suggestions": {"rich_text": self.notion.rich_text(edge_json)},
                "Graph Link Status": self._ki_prop("Graph Link Status", "linked-ai"),
            },
        )

    def _update_ki_cross_paper(
        self, ki_page_id: str, link_result: CrossPaperLinkResult
    ) -> None:
        all_proposals = link_result.proposals
        if not all_proposals:
            logger.debug("KI page %s: no cross-paper edges produced — remaining 'unlinked'.", ki_page_id)
            return

        auto_proposals = [p for p in all_proposals if p.channel == "auto"]
        suggest_proposals = [p for p in all_proposals if p.channel == "suggest"]

        if auto_proposals:
            payload = {"proposals": [p.model_dump() for p in auto_proposals]}
            edge_json = json.dumps(payload, ensure_ascii=False)
            if len(edge_json) > NOTION_BLOCK_MAX_CHARS:
                trimmed = list(auto_proposals)
                while trimmed and len(
                    json.dumps({"proposals": [p.model_dump() for p in trimmed]}, ensure_ascii=False)
                ) > NOTION_BLOCK_MAX_CHARS:
                    trimmed.pop()
                edge_json = json.dumps(
                    {"proposals": [p.model_dump() for p in trimmed]}, ensure_ascii=False
                )
        else:
            edge_json = json.dumps({"proposals": []}, ensure_ascii=False)

        self.notion.update_page(
            page_id=ki_page_id,
            properties={
                "Edge Suggestions": {"rich_text": self.notion.rich_text(edge_json)},
                "Graph Link Status": self._ki_prop("Graph Link Status", "linked-ai"),
            },
        )

        edge_blocks = self._render_cross_paper_edges_blocks(
            auto_proposals=auto_proposals,
            suggest_proposals=suggest_proposals,
        )
        if edge_blocks:
            try:
                append_in_batches(self.notion, ki_page_id, edge_blocks)
            except Exception:
                logger.warning(
                    "KI page %s: failed to append cross-paper edge blocks — "
                    "edge JSON is still stored in Edge Suggestions property.",
                    ki_page_id,
                )

    def _render_cross_paper_edges_blocks(
        self,
        auto_proposals: list[EdgeProposal],
        suggest_proposals: list[EdgeProposal],
    ) -> list[dict]:
        if not auto_proposals and not suggest_proposals:
            return []

        blocks: list[dict] = [
            divider_block(),
            heading_block("Proposed Cross-Paper Edges"),
        ]

        if auto_proposals:
            blocks.append(heading_block("Auto-Created Edges"))
            for p in auto_proposals:
                text = (
                    f"✅ {p.relation_type} → {p.target_concept_title}"
                    f"   (confidence: {p.confidence:.0%})\n"
                    f"   Because: {p.justification}\n"
                    f"   Fields: {', '.join(p.driving_fields)}\n"
                    f"   Would be wrong if: {p.falsifiability or '(not specified)'}"
                )
                blocks.append({
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{"type": "text", "text": {"content": text[:2000]}}],
                        "icon": {"type": "emoji", "emoji": "✅"},
                        "color": "green_background",
                    },
                })

        pure_suggest = [p for p in suggest_proposals if not p.demoted_from_auto]
        if pure_suggest:
            blocks.append(heading_block("Suggested Connections — Your Decision"))
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{
                        "type": "text",
                        "text": {
                            "content": (
                                "These were not auto-created. To accept: create the "
                                "edge in the Edges DB manually and uncheck needs_review. "
                                "To reject: leave unchecked."
                            )
                        },
                        "annotations": {"italic": True, "color": "gray"},
                    }],
                },
            })
            for p in pure_suggest:
                text = (
                    f"💡 {p.relation_type} → {p.target_concept_title}"
                    f"   (confidence: {p.confidence:.0%})  [NOT auto-created]\n"
                    f"   Why GPT thinks this is interesting: {p.justification}\n"
                    f"   Would be wrong if: {p.falsifiability or '(not specified)'}\n"
                    f"   → [ ] Accept (create edge manually)   [ ] Reject"
                )
                blocks.append(todo_block(text))

        demoted = [p for p in suggest_proposals if p.demoted_from_auto]
        if demoted:
            blocks.append(heading_block("Demoted Edges"))
            for p in demoted:
                text = (
                    f"⬇️ {p.relation_type} → {p.target_concept_title}"
                    f"  [demoted from auto — unstable or failed validation]\n"
                    f"   (confidence: {p.confidence:.0%})  [NOT auto-created]\n"
                    f"   Why GPT thinks this is interesting: {p.justification}\n"
                    f"   Would be wrong if: {p.falsifiability or '(not specified)'}\n"
                    f"   → [ ] Accept (create edge manually)   [ ] Reject"
                )
                blocks.append(todo_block(text))

        return blocks

    def _review_checklist_blocks(self) -> list[dict]:
        return [
            heading_block("Review"),
            todo_block("1. Is the title correct? (edit Name, or fill Corrected Title property)"),
            todo_block("2. Is the formal statement correct? Check the Statement block below."),
            todo_block("3. Are the assumptions and variables correct?"),
            todo_block("4. Review proposed edges in Edge Suggestions property."),
            todo_block("5. Set verification_status → verified or rejected"),
            divider_block(),
        ]

    def patch_paper_page(self, paper_page_id: str, ki_page_ids: list[str]) -> None:
        if not ki_page_ids:
            return
        try:
            existing_blocks = self.notion.get_block_children(paper_page_id)
            for block in existing_blocks:
                if block.get("type") == "heading_2":
                    rt = block.get("heading_2", {}).get("rich_text", [])
                    text = "".join(seg.get("plain_text", "") for seg in rt)
                    if "Extracted Concepts" in text:
                        logger.debug(
                            "PaperPage %s: 'Extracted Concepts' heading already exists — skipping.",
                            paper_page_id,
                        )
                        return
            count = len(ki_page_ids)
            callout_text = (
                f"{count} concept(s) extracted into Knowledge Inbox. "
                "Filter KI by Source Paper to review."
            )
            blocks: list[dict] = [
                heading_block("Extracted Concepts"),
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{"type": "text", "text": {"content": callout_text[:2000]}}],
                        "icon": {"type": "emoji", "emoji": "📚"},
                        "color": "blue_background",
                    },
                },
            ]
            append_in_batches(self.notion, paper_page_id, blocks)
            logger.info(
                "PaperPage %s: appended Extracted Concepts section (%d concept(s)).",
                paper_page_id, count,
            )
        except Exception:
            logger.warning(
                "PaperPage %s: could not patch paper page body — continuing.", paper_page_id
            )
