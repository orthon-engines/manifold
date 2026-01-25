"""
PRISM Sanity Checks - Data Validation with Humor

Validates data and returns helpful (sometimes funny) error messages.

"Your pressure sensor appears to be measuring the vacuum of space.
 Either you're on Mars, or something's wrong."
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import polars as pl


class Severity(Enum):
    """Issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Issue:
    """A detected data issue."""
    column: Optional[str]
    message: str
    severity: Severity
    suggestion: Optional[str] = None

    def __str__(self):
        prefix = f"[{self.severity.value.upper()}]"
        col = f" ({self.column})" if self.column else ""
        sug = f" Suggestion: {self.suggestion}" if self.suggestion else ""
        return f"{prefix}{col} {self.message}{sug}"


@dataclass
class SanityResult:
    """Result of sanity checks."""
    valid: bool
    issues: List[Issue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity in (Severity.ERROR, Severity.CRITICAL)]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


def validate(
    data: Union[pl.DataFrame, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> SanityResult:
    """
    Validate data and return issues found.

    Checks for:
    - Missing values
    - Constant columns (no variation)
    - Impossible physical values
    - Unit mismatches
    - Duplicate rows
    """
    issues = []
    stats = {}

    # Handle dict input
    if isinstance(data, dict):
        df = _dict_to_df(data)
    else:
        df = data

    stats['rows'] = len(df)
    stats['columns'] = len(df.columns)

    # Check for empty data
    if len(df) == 0:
        issues.append(Issue(
            column=None,
            message="Dataset is empty. Not even a single row. Impressive.",
            severity=Severity.CRITICAL,
            suggestion="Provide some actual data",
        ))
        return SanityResult(valid=False, issues=issues, stats=stats)

    # Check each column
    for col in df.columns:
        series = df[col]

        # Skip non-numeric
        if series.dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            continue

        # Missing values
        null_count = series.null_count()
        if null_count > 0:
            pct = 100 * null_count / len(series)
            severity = Severity.ERROR if pct > 50 else Severity.WARNING
            msg = f"{null_count} missing values ({pct:.1f}%)"
            if pct > 90:
                msg += " - this column is basically empty"
            issues.append(Issue(column=col, message=msg, severity=severity))

        # Constant columns
        n_unique = series.drop_nulls().n_unique()
        if n_unique == 1:
            val = series.drop_nulls().first()
            issues.append(Issue(
                column=col,
                message=f"Constant value: {val}. This sensor might be unplugged.",
                severity=Severity.WARNING,
                suggestion="Check sensor connection or remove column",
            ))
        elif n_unique == 2:
            issues.append(Issue(
                column=col,
                message="Only 2 unique values - might be a flag, not a measurement",
                severity=Severity.INFO,
            ))

        # Physical impossibilities
        _check_physical_bounds(col, series, issues)

    # Check for duplicates
    n_duplicates = len(df) - df.unique().height
    if n_duplicates > 0:
        pct = 100 * n_duplicates / len(df)
        issues.append(Issue(
            column=None,
            message=f"{n_duplicates} duplicate rows ({pct:.1f}%)",
            severity=Severity.WARNING if pct < 10 else Severity.ERROR,
        ))

    valid = not any(i.severity in (Severity.ERROR, Severity.CRITICAL) for i in issues)

    return SanityResult(valid=valid, issues=issues, stats=stats)


def _dict_to_df(data: Dict[str, Any]) -> pl.DataFrame:
    """Convert dict to DataFrame."""
    columns = {}
    for key, val in data.items():
        if hasattr(val, 'to_list'):
            columns[key] = val.to_list()
        elif hasattr(val, 'tolist'):
            columns[key] = val.tolist()
        elif isinstance(val, list):
            columns[key] = val
        else:
            columns[key] = [val]
    return pl.DataFrame(columns)


def _check_physical_bounds(col: str, series: pl.Series, issues: List[Issue]):
    """Check for physically impossible values."""
    col_lower = col.lower()

    # Temperature checks
    if any(t in col_lower for t in ['temp', 'degc', 'degf', 'kelvin']):
        min_val = series.min()
        max_val = series.max()

        if 'kelvin' in col_lower or col_lower.endswith('_k'):
            if min_val is not None and min_val < 0:
                issues.append(Issue(
                    column=col,
                    message=f"Temperature below absolute zero ({min_val} K). Physics would like a word.",
                    severity=Severity.ERROR,
                ))
        elif 'degc' in col_lower:
            if min_val is not None and min_val < -273.15:
                issues.append(Issue(
                    column=col,
                    message=f"Temperature below absolute zero ({min_val} C). That's not how thermodynamics works.",
                    severity=Severity.ERROR,
                ))

    # Pressure checks
    if any(p in col_lower for p in ['pressure', 'psi', 'bar', 'pa']):
        min_val = series.min()
        if min_val is not None and min_val < 0:
            if 'psia' in col_lower or 'absolute' in col_lower:
                issues.append(Issue(
                    column=col,
                    message=f"Negative absolute pressure ({min_val}). Vacuum harder than space.",
                    severity=Severity.ERROR,
                ))
            else:
                issues.append(Issue(
                    column=col,
                    message=f"Negative pressure ({min_val}). Could be gauge pressure, or a problem.",
                    severity=Severity.INFO,
                ))

    # Efficiency checks
    if 'efficiency' in col_lower or 'eta' in col_lower:
        min_val = series.min()
        max_val = series.max()
        if min_val is not None and min_val < 0:
            issues.append(Issue(
                column=col,
                message=f"Negative efficiency ({min_val}). Your machine is somehow making things worse.",
                severity=Severity.WARNING,
            ))
        if max_val is not None and max_val > 1.0:
            issues.append(Issue(
                column=col,
                message=f"Efficiency > 100% ({max_val}). Perpetual motion machine detected.",
                severity=Severity.WARNING,
                suggestion="Check if this should be a percentage (multiply by 100) or a ratio (0-1)",
            ))

    # Speed checks
    if 'rpm' in col_lower:
        max_val = series.max()
        if max_val is not None and max_val > 500000:
            issues.append(Issue(
                column=col,
                message=f"RPM of {max_val}. That's faster than a dental drill on steroids.",
                severity=Severity.WARNING,
            ))


__all__ = ['validate', 'SanityResult', 'Issue', 'Severity']
