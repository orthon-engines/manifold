"""
Load Stage Orchestrator

PURE: Loads 00_load.sql, creates base views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class LoadStage(StageOrchestrator):
    """Load observations and create base view."""

    SQL_FILE = '00_load.sql'

    VIEWS = [
        'v_base',
        'v_schema_validation',
        'v_signal_inventory',
        'v_data_quality',
    ]

    DEPENDS_ON = []  # First stage, no dependencies

    # Column mapping: source -> canonical
    COLUMN_MAP = {
        'timestamp': 'I',      # index column alternatives
        'index': 'I',
        'time': 'I',
        'value': 'y',          # value column alternatives
    }

    def load_observations(self, path: str) -> None:
        """
        Load observations parquet into database.

        Handles column renaming to canonical schema:
          entity_id, signal_id, I (index), y (value)

        PURE: Just column aliasing, no computation.
        """
        # Get actual columns from file
        cols = self.conn.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()
        col_names = [c[0] for c in cols]

        # Build SELECT with renames
        select_parts = []
        for col in col_names:
            canonical = self.COLUMN_MAP.get(col, col)
            if canonical != col:
                select_parts.append(f'"{col}" AS {canonical}')
            else:
                select_parts.append(f'"{col}"')

        select_clause = ', '.join(select_parts)
        self.conn.execute(f"CREATE OR REPLACE TABLE observations AS SELECT {select_clause} FROM '{path}'")

    def get_row_count(self) -> int:
        """Return number of rows loaded."""
        return self.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    def get_signal_count(self) -> int:
        """Return number of distinct signals."""
        return self.conn.execute("SELECT COUNT(DISTINCT signal_id) FROM observations").fetchone()[0]
