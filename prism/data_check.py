"""
PRISM Data Check

Validates observations.parquet BEFORE any compute.
Reads validation rules from canonical PRISM_SCHEMA.yaml.

This is PRISM's gate - if it fails, PRISM aborts.

Usage:
    # In runner.py (automatic)
    from prism.data_check import check_data, abort_if_invalid
    abort_if_invalid(observations_path)

    # Standalone
    python -m prism.data_check observations.parquet

    # With schema path
    python -m prism.data_check observations.parquet --schema /path/to/PRISM_SCHEMA.yaml
"""

import polars as pl
from pathlib import Path
import sys
import yaml
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# DEFAULT SCHEMA LOCATIONS
# =============================================================================

SCHEMA_LOCATIONS = [
    Path(__file__).parent.parent / "shared" / "schema" / "PRISM_SCHEMA.yaml",
    Path(__file__).parent / "PRISM_SCHEMA.yaml",
    Path.home() / "prism" / "PRISM_SCHEMA.yaml",
    Path("/etc/prism/PRISM_SCHEMA.yaml"),
]


def find_schema() -> Optional[Path]:
    """Find the canonical schema file."""
    for loc in SCHEMA_LOCATIONS:
        if loc.exists():
            return loc
    return None


def load_schema(path: Optional[Path] = None) -> dict:
    """Load the canonical schema."""
    if path is None:
        path = find_schema()

    if path is None or not path.exists():
        # Fallback to embedded defaults
        return get_default_schema()

    with open(path) as f:
        return yaml.safe_load(f)


def get_default_schema() -> dict:
    """Embedded schema defaults if file not found."""
    return {
        "columns": {
            "entity_id": {"type": "string", "nullable": False},
            "I": {"type": "uint32", "nullable": False},
            "signal_id": {"type": "string", "nullable": False},
            "value": {"type": "float64", "nullable": True},
        },
        "entity_requirements": {
            "min_signals_per_entity": 2,
            "min_observations_per_signal": 50,
        },
        "validation_rules": []
    }


# =============================================================================
# RESULT CLASS
# =============================================================================

@dataclass
class CheckResult:
    """Result of data validation."""
    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def error(self, rule_id: str, message: str):
        self.passed = False
        self.errors.append({"rule": rule_id, "message": message})

    def warn(self, rule_id: str, message: str):
        self.warnings.append({"rule": rule_id, "message": message})

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


# =============================================================================
# TYPE MAPPING
# =============================================================================

TYPE_MAP = {
    "string": [pl.String, pl.Utf8],
    "uint32": [pl.UInt32, pl.UInt64, pl.Int32, pl.Int64],
    "float64": [pl.Float64, pl.Float32],
    "integer": [pl.UInt32, pl.UInt64, pl.Int32, pl.Int64],
    "float": [pl.Float64, pl.Float32],
}


# =============================================================================
# MAIN CHECK FUNCTION
# =============================================================================

def check_data(
    path: Path,
    schema_path: Optional[Path] = None,
    verbose: bool = True
) -> CheckResult:
    """
    Validate observations.parquet against canonical schema.

    Args:
        path: Path to observations.parquet
        schema_path: Optional path to PRISM_SCHEMA.yaml
        verbose: Print detailed output

    Returns:
        CheckResult with passed/failed status
    """

    result = CheckResult()
    schema = load_schema(schema_path)

    # =========================================================================
    # HEADER
    # =========================================================================

    if verbose:
        print(f"\n{'='*60}")
        print(f"PRISM DATA CHECK")
        print(f"{'='*60}")
        print(f"File: {path}")
        if schema_path:
            print(f"Schema: {schema_path}")
        print()

    # =========================================================================
    # FILE EXISTS
    # =========================================================================

    if not path.exists():
        result.error("FILE_EXISTS", f"File not found: {path}")
        if verbose:
            print(f"  File not found: {path}")
        return result

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    try:
        df = pl.read_parquet(path)
    except Exception as e:
        result.error("FILE_READABLE", f"Failed to read parquet: {e}")
        if verbose:
            print(f"  Failed to read: {e}")
        return result

    result.stats["rows"] = df.shape[0]
    result.stats["columns"] = list(df.columns)

    if verbose:
        print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"Columns: {df.columns}\n")

    # =========================================================================
    # CHECK: REQUIRED COLUMNS EXIST
    # =========================================================================

    if verbose:
        print("[1] REQUIRED COLUMNS")

    required_cols = list(schema.get("columns", {}).keys())

    for col in required_cols:
        if col not in df.columns:
            result.error("COLUMNS_EXIST", f"Missing required column: {col}")
            if verbose:
                print(f"    X {col}: MISSING")
        else:
            if verbose:
                print(f"    + {col}: present")

    if not result.passed:
        if verbose:
            print("\n[!] Cannot continue - missing required columns")
        return result

    # =========================================================================
    # CHECK: COLUMN TYPES
    # =========================================================================

    if verbose:
        print(f"\n[2] COLUMN TYPES")

    for col, spec in schema.get("columns", {}).items():
        expected_type = spec.get("type", "string")
        valid_types = TYPE_MAP.get(expected_type, [pl.String])
        actual_type = df[col].dtype

        if actual_type not in valid_types:
            result.error("COLUMN_TYPES", f"{col}: expected {expected_type}, got {actual_type}")
            if verbose:
                print(f"    X {col}: {actual_type} (expected {expected_type})")
        else:
            if verbose:
                print(f"    + {col}: {actual_type}")

    # =========================================================================
    # CHECK: NULL VALUES
    # =========================================================================

    if verbose:
        print(f"\n[3] NULL VALUES")

    for col, spec in schema.get("columns", {}).items():
        nullable = spec.get("nullable", True)
        null_count = df[col].null_count()

        if not nullable and null_count > 0:
            result.error("NO_NULLS_IN_KEYS", f"{col} has {null_count} null values")
            if verbose:
                print(f"    X {col}: {null_count} nulls (not allowed)")
        else:
            if verbose:
                status = "+" if null_count == 0 else "~"
                print(f"    {status} {col}: {null_count} nulls")

    # =========================================================================
    # CHECK: I IS SEQUENTIAL
    # =========================================================================

    if verbose:
        print(f"\n[4] INDEX SEQUENTIALITY")

    # Check if I is sequential per entity/signal
    seq_check = (
        df.group_by(["entity_id", "signal_id"])
        .agg([
            pl.col("I").min().alias("min_i"),
            pl.col("I").max().alias("max_i"),
            pl.len().alias("count")
        ])
        .with_columns(
            ((pl.col("max_i") - pl.col("min_i") + 1) == pl.col("count")).alias("is_sequential")
        )
    )

    non_sequential = seq_check.filter(~pl.col("is_sequential"))

    if len(non_sequential) > 0:
        result.error("I_SEQUENTIAL", f"{len(non_sequential)} entity/signal pairs have non-sequential I")
        if verbose:
            print(f"    X {len(non_sequential)} groups have non-sequential I")
            sample = non_sequential.head(3)
            for row in sample.iter_rows(named=True):
                print(f"       - {row['entity_id']}/{row['signal_id']}: I goes {row['min_i']}->{row['max_i']} but only {row['count']} values")
    else:
        if verbose:
            print(f"    + I is sequential for all entity/signal pairs")

    # Check I starts at 0
    min_i_check = df.group_by("entity_id").agg(pl.col("I").min().alias("min_i"))
    non_zero = min_i_check.filter(pl.col("min_i") != 0)

    if len(non_zero) > 0:
        result.error("I_STARTS_ZERO", f"{len(non_zero)} entities don't start at I=0")
        if verbose:
            print(f"    X {len(non_zero)} entities don't start at I=0")
    else:
        if verbose:
            print(f"    + I starts at 0 for all entities")

    # =========================================================================
    # CHECK: MINIMUM SIGNALS PER ENTITY
    # =========================================================================

    if verbose:
        print(f"\n[5] SIGNALS PER ENTITY")

    min_signals = schema.get("entity_requirements", {}).get("min_signals_per_entity", 2)

    signals_per_entity = (
        df.group_by("entity_id")
        .agg(pl.col("signal_id").n_unique().alias("n_signals"))
    )

    insufficient = signals_per_entity.filter(pl.col("n_signals") < min_signals)

    result.stats["n_entities"] = len(signals_per_entity)
    result.stats["n_signals"] = df["signal_id"].n_unique()
    result.stats["signals"] = df["signal_id"].unique().sort().to_list()

    if len(insufficient) > 0:
        result.error("MIN_SIGNALS", f"{len(insufficient)} entities have <{min_signals} signals")
        if verbose:
            print(f"    X {len(insufficient)} entities have <{min_signals} signals")
            print(f"       Pair engines will FAIL for these entities")
    else:
        if verbose:
            print(f"    + All {len(signals_per_entity)} entities have >={min_signals} signals")

    if verbose:
        print(f"    Signals: {result.stats['signals']}")

    # =========================================================================
    # CHECK: MINIMUM OBSERVATIONS
    # =========================================================================

    if verbose:
        print(f"\n[6] OBSERVATIONS PER ENTITY/SIGNAL")

    min_obs = schema.get("entity_requirements", {}).get("min_observations_per_signal", 50)

    obs_per_group = (
        df.group_by(["entity_id", "signal_id"])
        .agg(pl.len().alias("n_obs"))
    )

    insufficient_obs = obs_per_group.filter(pl.col("n_obs") < min_obs)

    result.stats["min_obs"] = obs_per_group["n_obs"].min()
    result.stats["max_obs"] = obs_per_group["n_obs"].max()
    result.stats["mean_obs"] = obs_per_group["n_obs"].mean()

    if len(insufficient_obs) > 0:
        result.warn("MIN_OBSERVATIONS", f"{len(insufficient_obs)} groups have <{min_obs} observations")
        if verbose:
            print(f"    ~ {len(insufficient_obs)} groups have <{min_obs} observations")
            print(f"       Some engines may return NaN")
    else:
        if verbose:
            print(f"    + All groups have >={min_obs} observations")

    if verbose:
        print(f"    Range: {result.stats['min_obs']} - {result.stats['max_obs']} (mean: {result.stats['mean_obs']:.0f})")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    if verbose:
        print(f"\n{'='*60}")
        if result.passed:
            print("+ DATA CHECK PASSED")
            print("   PRISM will proceed with compute")
        else:
            print("X DATA CHECK FAILED")
            print(f"   {len(result.errors)} error(s):")
            for err in result.errors:
                print(f"   - [{err['rule']}] {err['message']}")
            print("\n[!] PRISM will NOT run on this data")
            print("   Fix the errors and try again")

        if result.warnings:
            print(f"\n   {len(result.warnings)} warning(s):")
            for warn in result.warnings:
                print(f"   - [{warn['rule']}] {warn['message']}")

        print(f"{'='*60}\n")

    return result


def abort_if_invalid(path: Path, schema_path: Optional[Path] = None):
    """
    Check data and abort if invalid.
    Use at the start of PRISM runner.
    """
    result = check_data(path, schema_path, verbose=True)

    if not result.passed:
        print("\n[!] ABORTING: Data validation failed")
        print("   Run ORTHON data_confirmation.py to fix the data")
        sys.exit(1)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PRISM Data Check")
    parser.add_argument("path", type=Path, help="Path to observations.parquet")
    parser.add_argument("--schema", type=Path, default=None, help="Path to PRISM_SCHEMA.yaml")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--json", action="store_true", help="Output JSON result")

    args = parser.parse_args()

    result = check_data(args.path, args.schema, verbose=not args.quiet and not args.json)

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))

    sys.exit(0 if result.passed else 1)
