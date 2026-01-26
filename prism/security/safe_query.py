"""
PRISM Safe Query Utilities
==========================

Prevents SQL injection and validates identifiers.
"""

import re
from pathlib import Path
from typing import Optional


class QueryValidationError(Exception):
    """Raised when query validation fails."""
    pass


# Valid identifier pattern: starts with letter/underscore, contains only alphanumeric/underscore
IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

# Suspicious SQL patterns
INJECTION_PATTERNS = [
    r";\s*DROP",
    r";\s*DELETE",
    r";\s*INSERT",
    r";\s*UPDATE",
    r";\s*CREATE",
    r";\s*ALTER",
    r"--",
    r"/\*",
    r"\*/",
    r"UNION\s+SELECT",
    r"OR\s+1\s*=\s*1",
    r"'\s*OR\s*'",
    r";\s*EXEC",
    r"xp_",
]


def safe_table_name(name: str) -> str:
    """
    Validate table name - alphanumeric and underscore only.

    Args:
        name: Table name to validate

    Returns:
        Validated table name

    Raises:
        QueryValidationError: If name is invalid
    """
    if not name:
        raise QueryValidationError("Empty table name")

    if not IDENTIFIER_PATTERN.match(name):
        raise QueryValidationError(
            f"Invalid table name: {name}. "
            "Must start with letter/underscore, contain only alphanumeric/underscore."
        )

    # Check length
    if len(name) > 128:
        raise QueryValidationError(f"Table name too long: {len(name)} chars (max 128)")

    return name


def safe_column_name(name: str) -> str:
    """
    Validate column name - alphanumeric and underscore only.

    Args:
        name: Column name to validate

    Returns:
        Validated column name

    Raises:
        QueryValidationError: If name is invalid
    """
    if not name:
        raise QueryValidationError("Empty column name")

    if not IDENTIFIER_PATTERN.match(name):
        raise QueryValidationError(
            f"Invalid column name: {name}. "
            "Must start with letter/underscore, contain only alphanumeric/underscore."
        )

    # Check length
    if len(name) > 128:
        raise QueryValidationError(f"Column name too long: {len(name)} chars (max 128)")

    return name


def validate_path_for_query(path: str) -> str:
    """
    Validate a file path for use in SQL queries.

    This is for DuckDB/Polars queries that reference parquet files.
    The path should be validated by safe_path() first, this adds
    SQL-specific checks.

    Args:
        path: File path to validate

    Returns:
        Validated path string

    Raises:
        QueryValidationError: If path contains SQL injection patterns
    """
    if not path:
        raise QueryValidationError("Empty path")

    # Convert to string and check
    path_str = str(path)

    # Check for injection patterns
    path_upper = path_str.upper()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, path_upper, re.IGNORECASE):
            raise QueryValidationError(f"Suspicious pattern in path: {pattern}")

    # Check for shell metacharacters
    shell_chars = ['`', '$', '|', '&', ';', '\n', '\r']
    for char in shell_chars:
        if char in path_str:
            raise QueryValidationError(f"Invalid character in path: {repr(char)}")

    # Verify it looks like a path
    try:
        p = Path(path_str)
        # Should have a valid extension for data files
        if p.suffix.lower() not in {'.parquet', '.csv', '.json', '.yaml', '.yml', ''}:
            raise QueryValidationError(f"Unexpected file extension: {p.suffix}")
    except Exception as e:
        raise QueryValidationError(f"Invalid path: {e}")

    return path_str


def looks_like_injection(value: str) -> bool:
    """
    Check if a string value looks like SQL injection.

    Args:
        value: String to check

    Returns:
        True if suspicious, False otherwise
    """
    if not value:
        return False

    value_upper = value.upper()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, value_upper, re.IGNORECASE):
            return True

    return False
