"""baseline documents schema

Revision ID: 69d06f228c5f
Revises: 
Create Date: 2026-08-25 21:29:39.881059
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa



revision: str = '69d06f228c5f'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False
        ),
        sa.Column(
            "filename",
            sa.String(),
            nullable=False
        ),
        sa.Column(
            "file_type",
            sa.String(),
            nullable=True
        ),
        sa.Column(
            "file_size",
            sa.Integer(),
            nullable=True
        ),
        sa.Column(
            "status",
            sa.String(),
            nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=True
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name="documents_pkey"
        )
    )

    op.create_index(
        "ix_documents_id",
        "documents",
        ["id"],
        unique=False
    )


def downgrade() -> None:
    op.drop_index(
        "ix_documents_id",
        table_name="documents"
    )

    op.drop_table(
        "documents"
    )
