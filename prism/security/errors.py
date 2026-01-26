"""
PRISM Safe Error Handling
=========================

Prevents information disclosure through error messages.
"""

import logging
import traceback
import uuid
from functools import wraps
from typing import Callable, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class SafeError(Exception):
    """
    User-safe error with internal logging.

    Use this for errors that should be shown to users.
    Internal details are logged but not returned.
    """

    def __init__(self, user_message: str, internal_error: Optional[Exception] = None):
        """
        Create a safe error.

        Args:
            user_message: Message safe to show to users
            internal_error: Optional internal exception (logged only)
        """
        self.user_message = user_message
        self.error_id = str(uuid.uuid4())[:8]

        # Log internal details
        if internal_error:
            logger.error(
                f"Error {self.error_id}: {user_message}\n"
                f"Internal: {internal_error}\n"
                f"{traceback.format_exc()}"
            )
        else:
            logger.error(f"Error {self.error_id}: {user_message}")

        super().__init__(user_message)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "error": self.user_message,
            "error_id": self.error_id,  # For support reference
        }


# Mapping of internal exceptions to user-safe messages
SAFE_MESSAGES = {
    FileNotFoundError: "File not found",
    PermissionError: "Access denied",
    ValueError: "Invalid input",
    TypeError: "Invalid data type",
    KeyError: "Missing required field",
    MemoryError: "Operation too large - try with smaller data",
    TimeoutError: "Operation timed out",
}


def get_safe_message(error: Exception) -> str:
    """
    Get a user-safe message for an exception.

    Args:
        error: The exception

    Returns:
        User-safe error message
    """
    # Check for known types
    for error_type, message in SAFE_MESSAGES.items():
        if isinstance(error, error_type):
            return message

    # Default generic message
    return "An error occurred"


def safe_error_handler(func: Callable) -> Callable:
    """
    Decorator to catch exceptions and return safe error responses.

    For use with FastAPI endpoints.

    Usage:
        @app.post("/endpoint")
        @safe_error_handler
        async def my_endpoint(...):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except SafeError as e:
            # Already safe, just return
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=e.to_dict())
        except FileNotFoundError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail={"error": "File not found"})
        except PermissionError:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail={"error": "Access denied"})
        except Exception as e:
            # Log full error internally
            error_id = str(uuid.uuid4())[:8]
            logger.error(f"Unhandled error {error_id}: {e}\n{traceback.format_exc()}")

            # Return generic message to user
            from fastapi import HTTPException
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal error",
                    "error_id": error_id,
                }
            )

    return wrapper


def log_error(error: Exception, context: str = "") -> str:
    """
    Log an error and return an error ID for reference.

    Args:
        error: The exception to log
        context: Optional context string

    Returns:
        Error ID for support reference
    """
    error_id = str(uuid.uuid4())[:8]
    logger.error(
        f"Error {error_id}"
        + (f" ({context})" if context else "")
        + f": {error}\n{traceback.format_exc()}"
    )
    return error_id
