"""description embedding

Revision ID: 957e5f52208f
Revises: 0610b1b70adf
Create Date: 2026-05-07 11:32:23.984882
"""
from typing import Sequence, Union
from alembic import op


revision: str = '957e5f52208f'
down_revision: Union[str, None] = '0610b1b70adf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE scene_observations
        ADD COLUMN description_embedding vector(384);
    """)
    op.execute("""
        CREATE INDEX idx_scene_obs_description_embedding
        ON scene_observations USING hnsw (description_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_scene_obs_description_embedding;")
    op.execute("""
        ALTER TABLE scene_observations
        DROP COLUMN IF EXISTS description_embedding;
    """)
