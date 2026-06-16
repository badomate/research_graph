"""Phase 1 — organization, selective parse/analysis, AI-suggestion quarantine.

Additive and idempotent: new tables are created with ``checkfirst=True`` (so it
is safe whether or not the app's ``create_all`` already built them), and the new
``projects`` columns are added only if missing. This lets an existing database be
brought up to the Phase-1 schema with ``alembic upgrade head`` without conflicts.

Revision ID: 0001_phase1
Revises:
Create Date: 2026-06-16
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlmodel import SQLModel

# Register all model tables on SQLModel.metadata.
from modules.store import models  # noqa: F401

revision = "0001_phase1"
down_revision = None
branch_labels = None
depends_on = None

NEW_TABLES = [
    "tags",
    "collections",
    "paper_tags",
    "paper_collections",
    "paper_projects",
    "parse_scopes",
    "parse_jobs",
    "paper_artifacts",
    "paper_chunks",
    "analysis_jobs",
    "ai_suggestions",
    "math_objects",
    "math_object_projects",
]

NEW_PROJECT_COLUMNS = {
    "status": sa.Column("status", sa.String(), nullable=False, server_default="active"),
    "priority": sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
    "updated_at": sa.Column("updated_at", sa.DateTime(), nullable=True),
}


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = set(insp.get_table_names())

    # Create new tables (only the ones not already present).
    tables = [
        SQLModel.metadata.tables[name]
        for name in NEW_TABLES
        if name in SQLModel.metadata.tables and name not in existing
    ]
    if tables:
        SQLModel.metadata.create_all(bind, tables=tables)

    # Add new columns to the pre-existing ``projects`` table if missing.
    if "projects" in existing:
        present = {c["name"] for c in insp.get_columns("projects")}
        to_add = [col for name, col in NEW_PROJECT_COLUMNS.items() if name not in present]
        if to_add:
            with op.batch_alter_table("projects") as batch:
                for col in to_add:
                    batch.add_column(col)


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = set(insp.get_table_names())

    if "projects" in existing:
        present = {c["name"] for c in insp.get_columns("projects")}
        with op.batch_alter_table("projects") as batch:
            for name in NEW_PROJECT_COLUMNS:
                if name in present:
                    batch.drop_column(name)

    for name in reversed(NEW_TABLES):
        if name in existing:
            op.drop_table(name)
