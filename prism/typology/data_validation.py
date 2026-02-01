"""
ORTHON Data Validation - The First Gate

Before ANY typology runs, ORTHON must validate the data contract.
Bad data is ORTHON's problem to fix, not PRISM's problem to accommodate.

Checks:
    1. File loads? — Can we read it at all
    2. Values exist? — Not all NaN, not empty, not constant-zero
    3. I is sequential? — 0,1,2,3... per signal_id, no gaps, no timestamps
    4. signal_id assigned? — Every row has one, no blanks
    5. Basic sanity — Row count, value range, infinities

If any check fails: STOP. Do not proceed. Return error to ORTHON.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class ValidationStatus(Enum):
    """Overall validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    check: str
    status: ValidationStatus
    message: str
    details: Optional[dict] = None


@dataclass
class ValidationResult:
    """Complete validation result."""
    status: ValidationStatus
    file_path: str
    issues: List[ValidationIssue] = field(default_factory=list)

    # Summary stats (populated if file loads)
    n_rows: int = 0
    n_signals: int = 0
    n_units: int = 0
    signals: List[str] = field(default_factory=list)

    # Value stats
    value_min: float = np.nan
    value_max: float = np.nan
    value_mean: float = np.nan
    n_nan: int = 0
    n_inf: int = 0

    def add_issue(self, check: str, status: ValidationStatus, message: str, details: dict = None):
        self.issues.append(ValidationIssue(check, status, message, details))
        if status == ValidationStatus.FAILED:
            self.status = ValidationStatus.FAILED
        elif status == ValidationStatus.WARNING and self.status != ValidationStatus.FAILED:
            self.status = ValidationStatus.WARNING

    def has_failures(self) -> bool:
        return any(i.status == ValidationStatus.FAILED for i in self.issues)

    def __str__(self) -> str:
        lines = [
            f"=== Data Validation: {self.status.value.upper()} ===",
            f"File: {self.file_path}",
        ]
        if self.n_rows > 0:
            lines.append(f"Rows: {self.n_rows:,} | Signals: {self.n_signals} | Units: {self.n_units}")
            lines.append(f"Values: min={self.value_min:.4g}, max={self.value_max:.4g}, mean={self.value_mean:.4g}")
            if self.n_nan > 0 or self.n_inf > 0:
                lines.append(f"Issues: {self.n_nan:,} NaN, {self.n_inf:,} Inf")

        lines.append("")
        for issue in self.issues:
            icon = "✓" if issue.status == ValidationStatus.PASSED else "✗" if issue.status == ValidationStatus.FAILED else "⚠"
            lines.append(f"  {icon} [{issue.check}] {issue.message}")

        return "\n".join(lines)


def _load_file(file_path: str) -> Tuple[Optional[object], Optional[str]]:
    """
    Attempt to load a data file.

    Supports: .parquet, .csv, .npz, .npy

    Returns:
        (data, error_message) - data is None if load failed
    """
    path = Path(file_path)

    if not path.exists():
        return None, f"File not found: {file_path}"

    suffix = path.suffix.lower()

    try:
        if suffix == '.parquet':
            import polars as pl
            return pl.read_parquet(file_path), None

        elif suffix == '.csv':
            import polars as pl
            # Try to auto-detect delimiter
            return pl.read_csv(file_path, infer_schema_length=10000), None

        elif suffix == '.npz':
            data = np.load(file_path)
            return data, None

        elif suffix == '.npy':
            data = np.load(file_path)
            return data, None

        else:
            return None, f"Unsupported file format: {suffix}"

    except Exception as e:
        return None, f"Failed to load file: {str(e)}"


def _check_required_columns(df, result: ValidationResult) -> bool:
    """Check that required columns exist."""
    required = ['signal_id', 'I', 'value']
    missing = [col for col in required if col not in df.columns]

    if missing:
        result.add_issue(
            "required_columns",
            ValidationStatus.FAILED,
            f"Missing required columns: {missing}",
            {"missing": missing, "available": list(df.columns)}
        )
        return False

    result.add_issue(
        "required_columns",
        ValidationStatus.PASSED,
        f"All required columns present: {required}"
    )
    return True


def _check_values_exist(df, result: ValidationResult) -> bool:
    """Check that values are not all NaN, empty, or constant-zero."""
    import polars as pl

    values = df['value']
    n_total = len(values)

    # Count issues
    n_null = values.null_count()
    n_nan = values.filter(values.is_nan()).len() if values.dtype in [pl.Float32, pl.Float64] else 0
    n_inf = values.filter(values.is_infinite()).len() if values.dtype in [pl.Float32, pl.Float64] else 0

    result.n_nan = n_null + n_nan
    result.n_inf = n_inf

    # All null/NaN?
    if n_null + n_nan >= n_total:
        result.add_issue(
            "values_exist",
            ValidationStatus.FAILED,
            "All values are NULL or NaN - no data to process"
        )
        return False

    # Get valid values
    valid = values.drop_nulls()
    if values.dtype in [pl.Float32, pl.Float64]:
        valid = valid.filter(~valid.is_nan() & ~valid.is_infinite())

    if len(valid) == 0:
        result.add_issue(
            "values_exist",
            ValidationStatus.FAILED,
            "No valid (non-null, non-NaN, non-Inf) values"
        )
        return False

    # All zeros?
    if (valid == 0).all():
        result.add_issue(
            "values_exist",
            ValidationStatus.FAILED,
            "All values are zero - no signal to analyze"
        )
        return False

    # Compute stats
    result.value_min = float(valid.min())
    result.value_max = float(valid.max())
    result.value_mean = float(valid.mean())

    # Warnings for high NaN/Inf counts
    nan_pct = (n_null + n_nan) / n_total * 100
    if nan_pct > 10:
        result.add_issue(
            "values_exist",
            ValidationStatus.WARNING,
            f"{nan_pct:.1f}% of values are NULL/NaN ({n_null + n_nan:,} of {n_total:,})"
        )
    elif nan_pct > 0:
        result.add_issue(
            "values_exist",
            ValidationStatus.PASSED,
            f"Values present ({nan_pct:.1f}% NULL/NaN)"
        )
    else:
        result.add_issue(
            "values_exist",
            ValidationStatus.PASSED,
            "All values present (no NULL/NaN)"
        )

    if n_inf > 0:
        result.add_issue(
            "infinities",
            ValidationStatus.WARNING,
            f"{n_inf:,} infinite values detected"
        )

    return True


def _check_i_sequential(df, result: ValidationResult) -> bool:
    """
    Check that I is sequential (0,1,2,3...) per signal_id.

    Rules:
        - I must start at 0 for each signal_id (within each unit_id if present)
        - I must be sequential with no gaps
        - I must not look like timestamps (huge values)
    """
    import polars as pl

    has_unit_id = 'unit_id' in df.columns
    group_cols = ['unit_id', 'signal_id'] if has_unit_id else ['signal_id']

    issues = []

    # Check each group
    for group_key, group_df in df.group_by(group_cols):
        i_values = group_df['I'].sort()

        # Convert to list for checking
        i_list = i_values.to_list()

        if len(i_list) == 0:
            continue

        # Check starts at 0
        if i_list[0] != 0:
            # Check if it looks like a timestamp (large value)
            if i_list[0] > 1_000_000:
                issues.append({
                    "group": group_key,
                    "issue": "timestamps_not_index",
                    "first_I": i_list[0],
                    "message": f"I starts at {i_list[0]} - looks like a timestamp, not an index"
                })
            else:
                issues.append({
                    "group": group_key,
                    "issue": "not_zero_start",
                    "first_I": i_list[0],
                    "message": f"I starts at {i_list[0]}, should start at 0"
                })

        # Check sequential (no gaps)
        expected = list(range(i_list[0], i_list[0] + len(i_list)))
        if i_list != expected:
            # Find first gap
            for idx, (actual, exp) in enumerate(zip(i_list, expected)):
                if actual != exp:
                    issues.append({
                        "group": group_key,
                        "issue": "gap_in_sequence",
                        "position": idx,
                        "expected": exp,
                        "actual": actual,
                        "message": f"Gap at position {idx}: expected I={exp}, got I={actual}"
                    })
                    break

    if issues:
        # Summarize issues
        timestamp_issues = [i for i in issues if i['issue'] == 'timestamps_not_index']
        zero_issues = [i for i in issues if i['issue'] == 'not_zero_start']
        gap_issues = [i for i in issues if i['issue'] == 'gap_in_sequence']

        if timestamp_issues:
            result.add_issue(
                "i_sequential",
                ValidationStatus.FAILED,
                f"I contains timestamps instead of sequential indices ({len(timestamp_issues)} groups)",
                {"examples": timestamp_issues[:3]}
            )
            return False

        if zero_issues:
            result.add_issue(
                "i_sequential",
                ValidationStatus.FAILED,
                f"I does not start at 0 for {len(zero_issues)} signal groups",
                {"examples": zero_issues[:3]}
            )
            return False

        if gap_issues:
            result.add_issue(
                "i_sequential",
                ValidationStatus.FAILED,
                f"I has gaps in {len(gap_issues)} signal groups",
                {"examples": gap_issues[:3]}
            )
            return False

    result.add_issue(
        "i_sequential",
        ValidationStatus.PASSED,
        "I is sequential (0,1,2,...) for all signal groups"
    )
    return True


def _check_signal_id_assigned(df, result: ValidationResult) -> bool:
    """Check that every row has a signal_id (no blanks)."""
    import polars as pl

    signal_ids = df['signal_id']
    n_total = len(signal_ids)

    # Count nulls and empty strings
    n_null = signal_ids.null_count()
    n_empty = signal_ids.filter(signal_ids == "").len()
    n_blank = n_null + n_empty

    if n_blank > 0:
        result.add_issue(
            "signal_id_assigned",
            ValidationStatus.FAILED,
            f"{n_blank:,} rows have missing/blank signal_id",
            {"null": n_null, "empty": n_empty}
        )
        return False

    # Get unique signals
    unique_signals = signal_ids.unique().to_list()
    result.n_signals = len(unique_signals)
    result.signals = sorted(unique_signals)

    result.add_issue(
        "signal_id_assigned",
        ValidationStatus.PASSED,
        f"All rows have signal_id ({result.n_signals} unique signals)"
    )
    return True


def _check_basic_sanity(df, result: ValidationResult) -> bool:
    """Basic sanity checks: row count, reasonable ranges."""
    import polars as pl

    result.n_rows = len(df)

    # Row count
    if result.n_rows == 0:
        result.add_issue(
            "row_count",
            ValidationStatus.FAILED,
            "File is empty (0 rows)"
        )
        return False

    if result.n_rows < 10:
        result.add_issue(
            "row_count",
            ValidationStatus.WARNING,
            f"Very few rows ({result.n_rows}) - may not have enough data for analysis"
        )
    else:
        result.add_issue(
            "row_count",
            ValidationStatus.PASSED,
            f"{result.n_rows:,} rows"
        )

    # Unit count
    if 'unit_id' in df.columns:
        result.n_units = df['unit_id'].n_unique()
    else:
        result.n_units = 1

    # Value range sanity
    if abs(result.value_max) > 1e15 or abs(result.value_min) > 1e15:
        result.add_issue(
            "value_range",
            ValidationStatus.WARNING,
            f"Extreme value range: [{result.value_min:.2e}, {result.value_max:.2e}]"
        )

    return True


def validate_observations(file_path: str, verbose: bool = True) -> ValidationResult:
    """
    Validate an observations file before typology.

    This is the FIRST GATE. If validation fails, do not proceed.
    Bad data is ORTHON's problem to fix.

    Args:
        file_path: Path to observations file (.parquet, .csv, .npz)
        verbose: Print results

    Returns:
        ValidationResult with status and issues
    """
    result = ValidationResult(
        status=ValidationStatus.PASSED,
        file_path=file_path
    )

    # 1. Can we load the file?
    data, error = _load_file(file_path)

    if error:
        result.add_issue("file_load", ValidationStatus.FAILED, error)
        if verbose:
            print(result)
        return result

    result.add_issue("file_load", ValidationStatus.PASSED, "File loaded successfully")

    # Handle different data types
    import polars as pl

    if isinstance(data, pl.DataFrame):
        df = data
    elif isinstance(data, np.ndarray):
        # Convert numpy array to polars - assume simple format
        if data.ndim == 1:
            df = pl.DataFrame({
                'signal_id': ['signal_0'] * len(data),
                'I': list(range(len(data))),
                'value': data.tolist()
            })
        else:
            result.add_issue(
                "file_format",
                ValidationStatus.FAILED,
                "Multi-dimensional numpy array - expected 1D or DataFrame format"
            )
            if verbose:
                print(result)
            return result
    elif hasattr(data, 'files'):  # npz file
        # Get first array
        key = data.files[0]
        arr = data[key]
        if arr.ndim == 1:
            df = pl.DataFrame({
                'signal_id': ['signal_0'] * len(arr),
                'I': list(range(len(arr))),
                'value': arr.tolist()
            })
        else:
            result.add_issue(
                "file_format",
                ValidationStatus.FAILED,
                f"npz array '{key}' is {arr.ndim}D - expected 1D"
            )
            if verbose:
                print(result)
            return result
    else:
        result.add_issue(
            "file_format",
            ValidationStatus.FAILED,
            f"Unknown data type: {type(data)}"
        )
        if verbose:
            print(result)
        return result

    # 2. Check required columns
    if not _check_required_columns(df, result):
        if verbose:
            print(result)
        return result

    # 3. Check values exist
    _check_values_exist(df, result)

    # 4. Check I is sequential
    _check_i_sequential(df, result)

    # 5. Check signal_id assigned
    _check_signal_id_assigned(df, result)

    # 6. Basic sanity
    _check_basic_sanity(df, result)

    if verbose:
        print(result)

    return result


def validate_benchmark_file(file_path: str, verbose: bool = True) -> ValidationResult:
    """
    Validate a benchmark file (may not have signal_id/I columns).

    For raw benchmark files like .npz or simple .csv, we auto-generate
    the required columns if they're missing.
    """
    result = ValidationResult(
        status=ValidationStatus.PASSED,
        file_path=file_path
    )

    # Load file
    data, error = _load_file(file_path)

    if error:
        result.add_issue("file_load", ValidationStatus.FAILED, error)
        if verbose:
            print(result)
        return result

    result.add_issue("file_load", ValidationStatus.PASSED, "File loaded successfully")

    import polars as pl

    # Convert to standard format
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            n = len(data)
            result.n_rows = n
            result.n_signals = 1
            result.n_units = 1
            result.value_min = float(np.nanmin(data))
            result.value_max = float(np.nanmax(data))
            result.value_mean = float(np.nanmean(data))
            result.n_nan = int(np.isnan(data).sum())
            result.n_inf = int(np.isinf(data).sum())

            result.add_issue(
                "format",
                ValidationStatus.PASSED,
                f"1D array with {n:,} values"
            )
        else:
            result.add_issue(
                "format",
                ValidationStatus.WARNING,
                f"{data.ndim}D array shape {data.shape}"
            )

    elif hasattr(data, 'files'):  # npz
        result.add_issue(
            "format",
            ValidationStatus.PASSED,
            f"NPZ file with arrays: {data.files}"
        )
        # Get stats from first array
        arr = data[data.files[0]]
        result.n_rows = arr.size
        result.value_min = float(np.nanmin(arr))
        result.value_max = float(np.nanmax(arr))
        result.value_mean = float(np.nanmean(arr))

    elif isinstance(data, pl.DataFrame):
        result.n_rows = len(data)
        if 'value' in data.columns:
            vals = data['value'].drop_nulls()
            result.value_min = float(vals.min()) if len(vals) > 0 else np.nan
            result.value_max = float(vals.max()) if len(vals) > 0 else np.nan
            result.value_mean = float(vals.mean()) if len(vals) > 0 else np.nan
        result.add_issue(
            "format",
            ValidationStatus.PASSED,
            f"DataFrame with {result.n_rows:,} rows, columns: {data.columns}"
        )

    if verbose:
        print(result)

    return result


if __name__ == "__main__":
    # Test on existing data
    import sys

    if len(sys.argv) > 1:
        result = validate_observations(sys.argv[1])
    else:
        # Test on PRISM data directory
        test_files = [
            "/Users/jasonrudder/prism/data/observations.parquet",
            "/Users/jasonrudder/prism/data/benchmarks/periodic/cwru_bearing/Data/1772 RPM/1772_Normal.npz",
        ]
        for f in test_files:
            print(f"\n{'='*60}")
            if "benchmark" in f or ".npz" in f:
                validate_benchmark_file(f)
            else:
                validate_observations(f)
