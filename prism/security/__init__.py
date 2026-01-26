"""
PRISM Security Module
=====================

Provides security utilities for file validation, query safety, rate limiting, and error handling.
"""

from prism.security.file_validator import (
    validate_upload,
    safe_path,
    FileValidationError,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
)

from prism.security.safe_query import (
    safe_table_name,
    safe_column_name,
    validate_path_for_query,
    QueryValidationError,
)

from prism.security.rate_limit import (
    rate_limit,
    RateLimitError,
    MAX_REQUESTS_PER_MINUTE,
    JOB_TIMEOUT_SECONDS,
)

from prism.security.errors import (
    SafeError,
    safe_error_handler,
)

__all__ = [
    # File validation
    "validate_upload",
    "safe_path",
    "FileValidationError",
    "ALLOWED_EXTENSIONS",
    "MAX_FILE_SIZE",
    # Query safety
    "safe_table_name",
    "safe_column_name",
    "validate_path_for_query",
    "QueryValidationError",
    # Rate limiting
    "rate_limit",
    "RateLimitError",
    "MAX_REQUESTS_PER_MINUTE",
    "JOB_TIMEOUT_SECONDS",
    # Error handling
    "SafeError",
    "safe_error_handler",
]
