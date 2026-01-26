"""
PRISM Rate Limiting and Resource Protection
============================================

Prevents abuse and resource exhaustion.
"""

import time
import logging
from functools import wraps
from typing import Dict, Callable, Any

logger = logging.getLogger(__name__)

# Simple in-memory rate limiter
_request_times: Dict[str, list] = {}
_active_jobs: Dict[str, float] = {}

# Limits
MAX_REQUESTS_PER_MINUTE = 30
MAX_CONCURRENT_JOBS = 3
JOB_TIMEOUT_SECONDS = 600  # 10 minutes


class RateLimitError(Exception):
    """Raised when rate limit exceeded."""
    pass


class ConcurrencyError(Exception):
    """Raised when too many concurrent jobs."""
    pass


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


def rate_limit(
    key: str,
    max_requests: int = MAX_REQUESTS_PER_MINUTE,
    window_seconds: int = 60
) -> None:
    """
    Check rate limit for key.

    Args:
        key: Rate limit key (e.g., IP address, API key)
        max_requests: Maximum requests allowed in window
        window_seconds: Window size in seconds

    Raises:
        RateLimitError: If rate limit exceeded
    """
    now = time.time()

    if key not in _request_times:
        _request_times[key] = []

    # Remove old entries
    _request_times[key] = [t for t in _request_times[key] if now - t < window_seconds]

    # Check limit
    if len(_request_times[key]) >= max_requests:
        wait_time = window_seconds - (now - _request_times[key][0])
        raise RateLimitError(
            f"Rate limit exceeded: {max_requests} requests per {window_seconds}s. "
            f"Try again in {wait_time:.0f}s."
        )

    # Record this request
    _request_times[key].append(now)


def check_concurrent_jobs(job_key: str, max_jobs: int = MAX_CONCURRENT_JOBS) -> None:
    """
    Check if we can start another concurrent job.

    Args:
        job_key: Unique job identifier
        max_jobs: Maximum concurrent jobs allowed

    Raises:
        ConcurrencyError: If too many concurrent jobs
    """
    now = time.time()

    # Clean up expired jobs
    expired = [k for k, t in _active_jobs.items() if now - t > JOB_TIMEOUT_SECONDS]
    for k in expired:
        logger.warning(f"Job {k} timed out after {JOB_TIMEOUT_SECONDS}s")
        del _active_jobs[k]

    # Check limit
    if len(_active_jobs) >= max_jobs:
        raise ConcurrencyError(
            f"Too many concurrent jobs: {len(_active_jobs)}/{max_jobs}. "
            "Please wait for existing jobs to complete."
        )

    # Register this job
    _active_jobs[job_key] = now


def complete_job(job_key: str) -> None:
    """
    Mark a job as complete.

    Args:
        job_key: Job identifier to remove
    """
    if job_key in _active_jobs:
        del _active_jobs[job_key]


def with_timeout(timeout_seconds: int = JOB_TIMEOUT_SECONDS):
    """
    Decorator to add timeout to a function.

    Note: Only works on Unix systems with SIGALRM support.

    Args:
        timeout_seconds: Timeout in seconds

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import signal
            import sys

            # Windows doesn't support SIGALRM
            if sys.platform == 'win32':
                logger.warning("Timeout not supported on Windows")
                return func(*args, **kwargs)

            def handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

            # Set alarm
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_seconds)

            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper
    return decorator


def get_rate_limit_status(key: str, window_seconds: int = 60) -> Dict[str, Any]:
    """
    Get current rate limit status for a key.

    Args:
        key: Rate limit key
        window_seconds: Window size

    Returns:
        Dict with requests_made, requests_remaining, reset_seconds
    """
    now = time.time()

    if key not in _request_times:
        return {
            "requests_made": 0,
            "requests_remaining": MAX_REQUESTS_PER_MINUTE,
            "reset_seconds": 0,
        }

    # Count recent requests
    recent = [t for t in _request_times[key] if now - t < window_seconds]
    requests_made = len(recent)
    requests_remaining = max(0, MAX_REQUESTS_PER_MINUTE - requests_made)

    # Time until oldest request expires
    reset_seconds = 0
    if recent:
        reset_seconds = max(0, window_seconds - (now - recent[0]))

    return {
        "requests_made": requests_made,
        "requests_remaining": requests_remaining,
        "reset_seconds": round(reset_seconds),
    }
