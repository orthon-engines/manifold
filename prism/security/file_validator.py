"""
PRISM File Upload Validation
============================

Whitelist-only file validation for security.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

# Allowed extensions
ALLOWED_EXTENSIONS = {'.parquet', '.csv', '.json', '.yaml', '.yml'}

# Magic bytes for file type validation
MAGIC_BYTES = {
    '.parquet': b'PAR1',  # Parquet magic at start and end
}

# Max file size: 500MB (industrial datasets can be large)
MAX_FILE_SIZE = 500 * 1024 * 1024


class FileValidationError(Exception):
    """Raised when file validation fails."""
    pass


def get_allowed_directories() -> list:
    """Get list of allowed data directories."""
    return [
        Path(os.path.expanduser("~/prism-mac/data")).resolve(),
        Path(os.path.expanduser("~/prism-inbox")).resolve(),
        Path("/tmp").resolve(),
    ]


def validate_upload(file_path: str, file_size: int = None) -> Tuple[bool, str]:
    """
    Validate uploaded file.

    Args:
        file_path: Path to the file
        file_size: Optional file size (will check if not provided)

    Returns:
        (is_valid, error_message)
    """
    path = Path(file_path)

    # 1. Check extension
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type not allowed: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    # 2. Check file exists
    if not path.exists():
        return False, "File does not exist"

    # 3. Check file size
    actual_size = file_size or path.stat().st_size
    if actual_size > MAX_FILE_SIZE:
        return False, f"File too large: {actual_size / 1024 / 1024:.1f}MB (max {MAX_FILE_SIZE / 1024 / 1024:.0f}MB)"

    # 4. Check magic bytes (for parquet)
    if ext == '.parquet':
        if not _check_parquet_magic(str(path)):
            return False, "Invalid parquet file (magic bytes mismatch)"

    # 5. Validate path is in allowed directory
    try:
        resolved = path.resolve()
        allowed_dirs = get_allowed_directories()
        in_allowed = any(
            str(resolved).startswith(str(d))
            for d in allowed_dirs
        )
        if not in_allowed:
            return False, "File path not in allowed directory"
    except Exception:
        return False, "Invalid file path"

    return True, ""


def _check_parquet_magic(file_path: str) -> bool:
    """Check parquet magic bytes at start and end of file."""
    try:
        with open(file_path, 'rb') as f:
            # Check start
            start = f.read(4)
            if start != b'PAR1':
                return False

            # Check end
            f.seek(-4, 2)  # 4 bytes from end
            end = f.read(4)
            if end != b'PAR1':
                return False

            return True
    except Exception:
        return False


def safe_path(user_input: str, base_dir: Optional[Path] = None) -> Path:
    """
    Convert user input to safe path within allowed directories.

    Args:
        user_input: User-provided path string
        base_dir: Optional base directory (must be in allowed list)

    Returns:
        Validated Path object

    Raises:
        FileValidationError: If path escapes sandbox or is invalid
    """
    if not user_input:
        raise FileValidationError("Empty path")

    # Resolve the path
    try:
        path = Path(user_input).resolve()
    except Exception as e:
        raise FileValidationError(f"Invalid path: {e}")

    # Check against allowed directories
    allowed_dirs = get_allowed_directories()
    if base_dir:
        base_resolved = Path(base_dir).resolve()
        if base_resolved not in allowed_dirs:
            raise FileValidationError("Base directory not allowed")
        allowed_dirs = [base_resolved]

    # Verify path is within allowed directory
    in_allowed = any(
        str(path).startswith(str(d))
        for d in allowed_dirs
    )

    if not in_allowed:
        raise FileValidationError(
            f"Path traversal detected or path not in allowed directories. "
            f"Allowed: {', '.join(str(d) for d in allowed_dirs)}"
        )

    return path


def validate_filename(filename: str) -> str:
    """
    Validate and sanitize a filename (no directory components).

    Args:
        filename: User-provided filename

    Returns:
        Sanitized filename

    Raises:
        FileValidationError: If filename is invalid
    """
    if not filename:
        raise FileValidationError("Empty filename")

    # Strip directory components
    clean = os.path.basename(filename)

    if not clean:
        raise FileValidationError("Invalid filename")

    # Check for suspicious patterns
    if clean.startswith('.'):
        raise FileValidationError("Hidden files not allowed")

    if '..' in clean:
        raise FileValidationError("Invalid filename")

    return clean
