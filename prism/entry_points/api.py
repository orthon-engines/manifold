"""
PRISM API - Compute interface for ORTHON.

ORTHON commands -> PRISM computes -> ORTHON SQL

Endpoints:
    POST /compute     - Run computation (synchronous, blocks until complete)
    GET  /health      - Status check
    GET  /files       - List available parquet files
    GET  /read        - Read parquet as JSON (query param: path)
    GET  /disciplines - List available disciplines

Security:
    - Path validation (sandbox to allowed directories)
    - Rate limiting (30 req/min default)
    - Safe error handling (no internal details leaked)
    - CORS restricted to localhost by default
"""

import logging
import os
import io
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Security imports
from prism.security.file_validator import (
    validate_upload,
    safe_path,
    FileValidationError,
    ALLOWED_EXTENSIONS,
)
from prism.security.rate_limit import (
    rate_limit,
    RateLimitError,
    check_concurrent_jobs,
    complete_job,
    ConcurrencyError,
    get_rate_limit_status,
)
from prism.security.errors import SafeError, log_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PRISM", version="0.4.0", description="Compute engine for ORTHON")

# CORS - Restrict to localhost and ORTHON origins
# Override with PRISM_CORS_ORIGINS env var (comma-separated)
ALLOWED_ORIGINS = os.environ.get(
    "PRISM_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# =============================================================================
# Models
# =============================================================================

class ComputeRequest(BaseModel):
    """Request from ORTHON to run computation.

    ORTHON sends:
        config: dict with 'discipline' key and computation parameters
        observations_path: path to observations.parquet
    """
    config: Dict[str, Any]
    observations_path: str


class ComputeResponse(BaseModel):
    """Response after computation completes.

    PRISM returns:
        status: 'complete' or 'error'
        results_path: directory containing output parquets
        parquets: list of parquet filenames created
        duration_seconds: computation time
        message: error message if status='error'
        hint: helpful hint for fixing errors
        engine: which engine failed (if error)
    """
    status: str  # 'complete' or 'error'
    results_path: Optional[str] = None
    parquets: Optional[List[str]] = None
    duration_seconds: Optional[float] = None
    message: Optional[str] = None
    hint: Optional[str] = None
    engine: Optional[str] = None
    error_id: Optional[str] = None  # For support reference


class JobStatus(BaseModel):
    """Status of a compute job."""
    job_id: str
    status: str  # pending, running, completed, failed
    discipline: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# State (in-memory for now, could be Redis/DB)
# =============================================================================

_jobs: Dict[str, JobStatus] = {}


# =============================================================================
# Helpers
# =============================================================================

def _get_data_dir() -> Path:
    """Get the PRISM data directory."""
    return Path(os.path.expanduser("~/prism-mac/data"))


def _get_inbox_dir() -> Path:
    """Get the PRISM inbox directory."""
    return Path(os.path.expanduser("~/prism-inbox"))


def _generate_job_id() -> str:
    """Generate a unique job ID."""
    return str(uuid.uuid4())[:8]


def _get_client_key(request: Request) -> str:
    """Get a key for rate limiting based on client."""
    # Use X-Forwarded-For if behind proxy, otherwise client host
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _validate_observations_path(path: str) -> Path:
    """
    Validate observations path is safe.

    Args:
        path: User-provided path

    Returns:
        Validated Path object

    Raises:
        HTTPException: If validation fails
    """
    try:
        validated = safe_path(path)

        # Must be a parquet file
        if validated.suffix.lower() != '.parquet':
            raise HTTPException(400, "observations_path must be a .parquet file")

        # Must exist
        if not validated.exists():
            raise HTTPException(404, f"File not found: {validated.name}")

        # Validate it's actually a parquet file
        is_valid, error = validate_upload(str(validated))
        if not is_valid:
            raise HTTPException(400, error)

        return validated

    except FileValidationError as e:
        logger.warning(f"Path validation failed: {e}")
        raise HTTPException(400, "Invalid file path")


def _run_compute_sync(config: Dict, observations_path: Path, job_id: str) -> ComputeResponse:
    """Run computation synchronously. Returns when complete."""
    start_time = time.time()

    discipline = config.get("discipline")
    output_dir = _get_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write config to YAML for PRISM to read
        import yaml
        config_path = output_dir / "config.yaml"

        # Build config with required fields + user overrides
        default_engines = {
            "vector": {
                "enabled": [
                    "hurst_dfa", "sample_entropy", "spectral_slope",
                    "garch", "rqa", "stationarity", "trend"
                ]
            },
            "geometry": {
                "enabled": [
                    "bg_correlation", "bg_distance", "bg_clustering"
                ]
            },
        }

        prism_config = {
            "discipline": discipline,
            "observations_path": str(observations_path),
            "window": config.get("window", {"size": 100, "stride": 50}),
            "min_samples": config.get("min_samples", 50),
            "engines": config.get("engines", default_engines),
            **{k: v for k, v in config.items() if k not in ["discipline", "window", "min_samples", "engines"]},
        }
        with open(config_path, 'w') as f:
            yaml.dump(prism_config, f)

        # Copy observations to data dir if not already there
        target_obs = output_dir / "observations.parquet"
        if observations_path != target_obs:
            import shutil
            shutil.copy(observations_path, target_obs)

        # Run PRISM compute
        cmd = [
            sys.executable, "-m", "prism.entry_points.compute",
            "--force",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
            env={**os.environ, "PRISM_DATA_DIR": str(output_dir)},
            timeout=600,  # 10 minute timeout
        )

        duration = time.time() - start_time

        # Find created parquet files
        parquets = [f.name for f in output_dir.glob("*.parquet")
                    if f.name != "observations.parquet"]

        core_created = any(p in parquets for p in ["vector.parquet", "physics.parquet"])

        if result.returncode != 0 and not core_created:
            # Log internal error
            error_id = log_error(
                Exception(result.stderr or result.stdout or "Unknown error"),
                f"compute job {job_id}"
            )

            # Return safe message
            hint = None
            if "Missing required" in (result.stderr or ""):
                hint = "Check config has all required constants for this discipline"

            return ComputeResponse(
                status="error",
                message="Computation failed",
                hint=hint,
                error_id=error_id,
                duration_seconds=round(duration, 2),
            )

        # Success
        status_file = output_dir / "job_status.json"
        with open(status_file, 'w') as f:
            json.dump({
                "status": "complete",
                "discipline": discipline,
                "timestamp": datetime.now().isoformat(),
            }, f)

        return ComputeResponse(
            status="complete",
            results_path=str(output_dir),
            parquets=parquets,
            duration_seconds=round(duration, 2),
        )

    except subprocess.TimeoutExpired:
        error_id = log_error(Exception("Timeout"), f"compute job {job_id}")
        return ComputeResponse(
            status="error",
            message="Computation timed out (10 minute limit)",
            error_id=error_id,
            duration_seconds=600,
        )
    except Exception as e:
        error_id = log_error(e, f"compute job {job_id}")
        return ComputeResponse(
            status="error",
            message="Internal error during computation",
            error_id=error_id,
            duration_seconds=round(time.time() - start_time, 2),
        )
    finally:
        complete_job(job_id)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    from prism import __version__
    return {
        "status": "ok",
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/disciplines")
async def list_disciplines(request: Request):
    """List available disciplines."""
    try:
        rate_limit(_get_client_key(request))
    except RateLimitError as e:
        raise HTTPException(429, str(e))

    from prism.disciplines import DISCIPLINES
    return {
        "disciplines": list(DISCIPLINES.keys()),
        "details": {k: v.get("name", k) for k, v in DISCIPLINES.items()},
    }


@app.get("/disciplines/{discipline}")
async def get_discipline(discipline: str, request: Request):
    """Get discipline requirements and engines."""
    try:
        rate_limit(_get_client_key(request))
    except RateLimitError as e:
        raise HTTPException(429, str(e))

    from prism.disciplines import DISCIPLINES
    from prism.disciplines.requirements import get_requirements_text

    if discipline not in DISCIPLINES:
        raise HTTPException(404, "Unknown discipline")

    return {
        "discipline": discipline,
        "info": DISCIPLINES[discipline],
        "requirements_text": get_requirements_text(discipline),
    }


@app.post("/compute", response_model=ComputeResponse)
async def compute(request_body: ComputeRequest, request: Request):
    """
    Run PRISM computation (synchronous).

    ORTHON sends:
        config: dict with 'discipline' and parameters
        observations_path: path to observations.parquet

    PRISM returns when complete:
        status: 'complete' or 'error'
        results_path: directory with output parquets
        parquets: list of created files
        duration_seconds: how long it took
    """
    client_key = _get_client_key(request)

    # Rate limit
    try:
        rate_limit(client_key, max_requests=10, window_seconds=60)
    except RateLimitError as e:
        raise HTTPException(429, str(e))

    # Concurrency limit
    job_id = _generate_job_id()
    try:
        check_concurrent_jobs(job_id)
    except ConcurrencyError as e:
        raise HTTPException(429, str(e))

    # Validate observations path (SECURITY)
    observations_path = _validate_observations_path(request_body.observations_path)

    # Validate discipline
    from prism.disciplines import DISCIPLINES
    discipline = request_body.config.get("discipline")
    if discipline and discipline not in DISCIPLINES:
        return ComputeResponse(
            status="error",
            message="Unknown discipline",
            hint=f"Available: {', '.join(DISCIPLINES.keys())}",
        )

    # Run synchronously
    return _run_compute_sync(request_body.config, observations_path, job_id)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    """Get job status."""
    try:
        rate_limit(_get_client_key(request))
    except RateLimitError as e:
        raise HTTPException(429, str(e))

    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


@app.get("/jobs")
async def list_jobs(request: Request):
    """List all jobs."""
    try:
        rate_limit(_get_client_key(request))
    except RateLimitError as e:
        raise HTTPException(429, str(e))

    return {"jobs": list(_jobs.values())}


@app.get("/files")
async def list_files(request: Request):
    """List available parquet files."""
    try:
        rate_limit(_get_client_key(request))
    except RateLimitError as e:
        raise HTTPException(429, str(e))

    data_dir = _get_data_dir()
    files = {}
    for f in ['observations', 'data', 'vector', 'geometry', 'dynamics', 'physics']:
        path = data_dir / f"{f}.parquet"
        if path.exists():
            files[f] = {
                "exists": True,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
        else:
            files[f] = {"exists": False}
    return files


@app.get("/read/{filename}")
async def read_file_by_name(filename: str, request: Request, limit: int = 100, offset: int = 0):
    """Read a parquet file by name and return as JSON."""
    try:
        rate_limit(_get_client_key(request))
    except RateLimitError as e:
        raise HTTPException(429, str(e))

    import polars as pl

    # Validate filename (no path traversal)
    if '/' in filename or '\\' in filename or '..' in filename:
        raise HTTPException(400, "Invalid filename")

    if not filename.endswith('.parquet'):
        filename = f"{filename}.parquet"

    # Only allow known files
    allowed_files = {
        'observations.parquet', 'data.parquet', 'vector.parquet',
        'geometry.parquet', 'dynamics.parquet', 'physics.parquet'
    }
    if filename not in allowed_files:
        raise HTTPException(400, "File not allowed")

    path = _get_data_dir() / filename
    if not path.exists():
        raise HTTPException(404, "File not found")

    # Limit pagination
    limit = min(limit, 1000)  # Max 1000 rows
    offset = max(offset, 0)

    df = pl.read_parquet(path)
    return {
        "file": filename,
        "total_rows": len(df),
        "columns": df.columns,
        "offset": offset,
        "limit": limit,
        "data": df.slice(offset, limit).to_dicts(),
    }


@app.get("/read")
async def read_file_by_path(
    request: Request,
    path: str = Query(..., description="Path to parquet file"),
    limit: int = 100,
    offset: int = 0
):
    """Read a parquet file by full path and return as JSON."""
    try:
        rate_limit(_get_client_key(request))
    except RateLimitError as e:
        raise HTTPException(429, str(e))

    import polars as pl

    # Validate path (SECURITY)
    try:
        validated_path = safe_path(path)
    except FileValidationError:
        raise HTTPException(400, "Invalid file path")

    if not validated_path.suffix.lower() == '.parquet':
        raise HTTPException(400, "Only .parquet files supported")

    if not validated_path.exists():
        raise HTTPException(404, "File not found")

    # Validate it's actually a parquet
    is_valid, error = validate_upload(str(validated_path))
    if not is_valid:
        raise HTTPException(400, "Invalid parquet file")

    # Limit pagination
    limit = min(limit, 1000)
    offset = max(offset, 0)

    df = pl.read_parquet(validated_path)
    return {
        "file": validated_path.name,
        "total_rows": len(df),
        "columns": df.columns,
        "offset": offset,
        "limit": limit,
        "data": df.slice(offset, limit).to_dicts(),
    }


@app.get("/download/{filename}")
async def download_file(filename: str, request: Request):
    """Download a parquet file."""
    try:
        rate_limit(_get_client_key(request), max_requests=5, window_seconds=60)
    except RateLimitError as e:
        raise HTTPException(429, str(e))

    # Validate filename
    if '/' in filename or '\\' in filename or '..' in filename:
        raise HTTPException(400, "Invalid filename")

    if not filename.endswith('.parquet'):
        filename = f"{filename}.parquet"

    # Only allow known files
    allowed_files = {
        'observations.parquet', 'data.parquet', 'vector.parquet',
        'geometry.parquet', 'dynamics.parquet', 'physics.parquet'
    }
    if filename not in allowed_files:
        raise HTTPException(400, "File not allowed")

    path = _get_data_dir() / filename
    if not path.exists():
        raise HTTPException(404, "File not found")

    return StreamingResponse(
        io.BytesIO(path.read_bytes()),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/rate-limit-status")
async def rate_limit_status(request: Request):
    """Get current rate limit status for this client."""
    client_key = _get_client_key(request)
    return get_rate_limit_status(client_key)


# Removed /trigger-github - external API calls should be handled by ORTHON


# =============================================================================
# CLI
# =============================================================================

def main():
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="PRISM API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    print("=" * 60)
    print("PRISM API - Compute Engine for ORTHON")
    print("=" * 60)
    print(f"Server:  http://{args.host}:{args.port}")
    print(f"Docs:    http://{args.host}:{args.port}/docs")
    print(f"Health:  http://{args.host}:{args.port}/health")
    print("=" * 60)
    print("\nSecurity enabled:")
    print("  - Path validation (sandbox)")
    print("  - Rate limiting (30 req/min)")
    print("  - Concurrency limit (3 jobs)")
    print("  - 10 minute job timeout")
    print("=" * 60)

    uvicorn.run(
        "prism.entry_points.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
