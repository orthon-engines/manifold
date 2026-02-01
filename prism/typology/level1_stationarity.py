"""
Level 1: Bachelor Typology — Stationarity Testing

Core Question: "Is this signal stationary, or not?"

Methods:
    - ADF (Augmented Dickey-Fuller): H0 = unit root exists (non-stationary)
    - KPSS: H0 = series IS stationary (opposite of ADF)
    - ACF decay: how fast does autocorrelation drop?
    - Variance ratio: does local variance change across the signal?

Decision Table (ADF x KPSS):

    | ADF rejects H0?  | KPSS rejects H0?  | Classification         |
    | (stationary)      | (non-stationary)   |                        |
    |-------------------|--------------------|------------------------|
    | Yes               | No                 | STATIONARY             |
    | No                | Yes                | NON_STATIONARY         |
    | Yes               | Yes                | DIFFERENCE_STATIONARY  |
    | No                | No                 | TREND_STATIONARY       |

    When both tests agree, the answer is clear.
    When they conflict, the label describes the recommended transform.

This is the FIRST GATE. Level 2 (Masters) builds on this result to
classify signal type (periodic, chaotic, random, etc).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List
import warnings
import numpy as np

from statsmodels.tsa.stattools import adfuller, kpss, acf


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StationarityType(Enum):
    """Stationarity classification from ADF + KPSS joint interpretation."""
    STATIONARY = "stationary"
    TREND_STATIONARY = "trend_stationary"
    DIFFERENCE_STATIONARY = "difference_stationary"
    NON_STATIONARY = "non_stationary"
    INSUFFICIENT_DATA = "insufficient_data"


class Confidence(Enum):
    """How much to trust the classification."""
    HIGH = "high"          # Both tests well inside thresholds
    MEDIUM = "medium"      # At least one test near the boundary
    LOW = "low"            # p-values at lookup-table limits
    UNKNOWN = "unknown"    # Test failed or insufficient data


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StationarityResult:
    """
    Complete Level 1 result.

    Designed so Level 2 can read everything it needs without re-running
    any tests. All raw values are preserved.
    """

    # --- Classification ---
    stationarity_type: StationarityType
    is_stationary: bool                   # True only when STATIONARY
    confidence: Confidence                # How sure are we?

    # --- ADF ---
    adf_statistic: float
    adf_pvalue: float
    adf_rejects: bool                     # True = rejects unit root = stationary evidence

    # --- KPSS (level, regression='c') ---
    kpss_statistic: float
    kpss_pvalue: float
    kpss_rejects: bool                    # True = rejects stationarity = non-stationary evidence
    kpss_pvalue_at_lower_bound: bool      # p reported as 0.01 (true p may be smaller)
    kpss_pvalue_at_upper_bound: bool      # p reported as 0.10 (true p may be larger)

    # --- KPSS trend (regression='ct'), only when needed ---
    kpss_ct_pvalue: Optional[float]       # None if not run
    kpss_ct_rejects: Optional[bool]       # None if not run

    # --- ACF characterisation ---
    acf_decay_lag: int                    # First lag where |ACF| < 1/e
    acf_half_life: int                    # First lag where |ACF| < 0.5
    acf_values: Optional[np.ndarray]      # Raw ACF array for Level 2

    # --- Variance ratio (segmented) ---
    variance_ratio: float                 # var(second_half) / var(first_half)
    variance_stable: bool                 # True if ratio is in [0.5, 2.0]

    # --- Metadata ---
    n_samples: int
    recommendation: str

    # --- For serialisation: drop numpy arrays ---
    def to_dict(self) -> dict:
        """Flat dictionary for parquet/JSON. Drops acf_values array."""
        d = {}
        for k, v in self.__dict__.items():
            if k == "acf_values":
                continue
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# ADF test
# ---------------------------------------------------------------------------

def compute_adf(
    y: np.ndarray,
    max_lag: Optional[int] = None,
) -> Tuple[float, float, bool]:
    """
    Augmented Dickey-Fuller test.

    H0: Series has a unit root (non-stationary).
    Reject H0 (p < 0.05) => evidence of stationarity.

    Returns (statistic, p_value, rejects_H0).
    """
    try:
        result = adfuller(y, maxlag=max_lag, autolag="AIC")
        stat = float(result[0])
        pval = float(result[1])
        return stat, pval, pval < 0.05
    except Exception:
        return np.nan, np.nan, False


# ---------------------------------------------------------------------------
# KPSS test (with warning capture)
# ---------------------------------------------------------------------------

def compute_kpss(
    y: np.ndarray,
    regression: str = "c",
) -> Tuple[float, float, bool, bool, bool]:
    """
    KPSS test.

    H0: Series IS stationary.
    Reject H0 (p < 0.05) => evidence of non-stationarity.

    Returns (statistic, p_value, rejects_H0,
             p_at_lower_bound, p_at_upper_bound).

    The KPSS lookup table only covers p in [0.01, 0.10].
    When the true p is outside that range, statsmodels clamps
    and issues an InterpolationWarning. We capture that.
    """
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = kpss(y, regression=regression, nlags="auto")

        stat = float(result[0])
        pval = float(result[1])

        # Detect clamped p-values from warning messages
        at_lower = False
        at_upper = False
        for w in caught:
            msg = str(w.message)
            if "smaller than the p-value returned" in msg:
                at_lower = True          # true p < 0.01
            elif "greater than the p-value returned" in msg:
                at_upper = True          # true p > 0.10

        return stat, pval, pval < 0.05, at_lower, at_upper
    except Exception:
        return np.nan, np.nan, False, False, False


# ---------------------------------------------------------------------------
# ACF decay
# ---------------------------------------------------------------------------

def compute_acf_decay(
    y: np.ndarray,
    max_lag: int = 100,
    return_values: bool = True,
) -> Tuple[int, int, Optional[np.ndarray]]:
    """
    ACF decay characteristics.

    Returns (decay_lag, half_life, acf_array).
        decay_lag : first lag where |ACF| < 1/e (~0.368)
        half_life : first lag where |ACF| < 0.5
        acf_array : raw ACF values (None if return_values=False)
    """
    try:
        nlags = min(max_lag, len(y) // 4)
        if nlags < 1:
            return -1, -1, None

        acf_vals = acf(y, nlags=nlags, fft=True)

        threshold_e = 1.0 / np.e
        decay_lag = nlags
        half_life = nlags

        for i in range(1, len(acf_vals)):
            if decay_lag == nlags and abs(acf_vals[i]) < threshold_e:
                decay_lag = i
            if half_life == nlags and abs(acf_vals[i]) < 0.5:
                half_life = i
            if decay_lag < nlags and half_life < nlags:
                break

        return decay_lag, half_life, (acf_vals if return_values else None)
    except Exception:
        return -1, -1, None


# ---------------------------------------------------------------------------
# Variance ratio (segmented stability check)
# ---------------------------------------------------------------------------

def compute_variance_ratio(y: np.ndarray) -> Tuple[float, bool]:
    """
    Compare variance of first half vs second half.

    A stationary signal should have roughly equal variance in both
    halves.  A ratio far from 1.0 suggests the signal's character
    is changing (even if ADF/KPSS don't catch it).

    Returns (ratio, is_stable).
        ratio = var(second_half) / var(first_half)
        is_stable = True if ratio is in [0.5, 2.0]
    """
    n = len(y)
    if n < 20:
        return np.nan, False

    mid = n // 2
    var1 = np.var(y[:mid])
    var2 = np.var(y[mid:])

    if var1 < 1e-15:
        # First half is essentially constant
        return np.inf if var2 > 1e-15 else 1.0, var2 < 1e-15

    ratio = float(var2 / var1)
    return ratio, 0.5 <= ratio <= 2.0


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def classify_stationarity(
    adf_rejects: bool,
    kpss_rejects: bool,
) -> StationarityType:
    """
    Joint ADF x KPSS classification.

    | ADF rejects? (stat.) | KPSS rejects? (non-stat.) | Result                |
    |----------------------|---------------------------|-----------------------|
    | Yes                  | No                        | STATIONARY            |
    | No                   | Yes                       | NON_STATIONARY        |
    | Yes                  | Yes                       | DIFFERENCE_STATIONARY |
    | No                   | No                        | TREND_STATIONARY      |
    """
    if adf_rejects and not kpss_rejects:
        return StationarityType.STATIONARY
    elif not adf_rejects and kpss_rejects:
        return StationarityType.NON_STATIONARY
    elif adf_rejects and kpss_rejects:
        return StationarityType.DIFFERENCE_STATIONARY
    else:
        return StationarityType.TREND_STATIONARY


def assess_confidence(
    adf_pvalue: float,
    kpss_pvalue: float,
    kpss_at_lower: bool,
    kpss_at_upper: bool,
) -> Confidence:
    """
    Estimate confidence based on how far p-values are from 0.05.

    HIGH   : both p-values well away from 0.05 and not clamped
    MEDIUM : one p-value near the boundary OR one clamped
    LOW    : both near boundary or both clamped
    """
    if np.isnan(adf_pvalue) or np.isnan(kpss_pvalue):
        return Confidence.UNKNOWN

    # "Near boundary" = within [0.01, 0.10] of the 0.05 threshold
    adf_clear = adf_pvalue < 0.01 or adf_pvalue > 0.10
    kpss_clear = kpss_pvalue < 0.01 or kpss_pvalue > 0.10

    clamped = kpss_at_lower or kpss_at_upper

    if adf_clear and kpss_clear and not clamped:
        return Confidence.HIGH
    elif adf_clear or kpss_clear:
        return Confidence.MEDIUM
    else:
        return Confidence.LOW


def get_recommendation(stype: StationarityType) -> str:
    """Action recommendation for each classification."""
    return {
        StationarityType.STATIONARY:
            "Signal is stationary. Proceed with standard features.",
        StationarityType.TREND_STATIONARY:
            "Deterministic trend detected. Detrend before features.",
        StationarityType.DIFFERENCE_STATIONARY:
            "Stochastic trend / heteroscedastic. Difference or use rolling windows.",
        StationarityType.NON_STATIONARY:
            "Non-stationary. Apply transformations before analysis.",
        StationarityType.INSUFFICIENT_DATA:
            "Fewer than 20 samples. Cannot test stationarity.",
    }.get(stype, "Unknown classification.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def test_stationarity(
    y: np.ndarray,
    verbose: bool = False,
) -> StationarityResult:
    """
    Level 1 typology: test signal stationarity.

    Pipeline:
        1. ADF  (H0: unit root)
        2. KPSS with regression='c'  (H0: level-stationary)
        3. ACF decay characterisation
        4. Variance-ratio stability check
        5. If both ADF & KPSS say non-stationary, try KPSS 'ct'
        6. Classify and assess confidence

    Args:
        y:  1-D array of signal values.
        verbose:  print results to stdout.

    Returns:
        StationarityResult with everything Level 2 needs.
    """
    # --- Clean input ---
    y = np.asarray(y, dtype=np.float64).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    # --- Guard: constant signal ---
    if np.ptp(y) < 1e-12:
        # All identical values. Trivially stationary, but ADF/KPSS will choke.
        r = StationarityResult(
            stationarity_type=StationarityType.STATIONARY,
            is_stationary=True,
            confidence=Confidence.HIGH,
            adf_statistic=np.nan,
            adf_pvalue=0.0,
            adf_rejects=True,
            kpss_statistic=np.nan,
            kpss_pvalue=1.0,
            kpss_rejects=False,
            kpss_pvalue_at_lower_bound=False,
            kpss_pvalue_at_upper_bound=False,
            kpss_ct_pvalue=None,
            kpss_ct_rejects=None,
            acf_decay_lag=0,
            acf_half_life=0,
            acf_values=None,
            variance_ratio=1.0,
            variance_stable=True,
            n_samples=n,
            recommendation="Constant signal. No variation to analyse.",
        )
        if verbose:
            _print_result(r)
        return r

    # --- Guard: too short ---
    if n < 20:
        r = StationarityResult(
            stationarity_type=StationarityType.INSUFFICIENT_DATA,
            is_stationary=False,
            confidence=Confidence.UNKNOWN,
            adf_statistic=np.nan,
            adf_pvalue=np.nan,
            adf_rejects=False,
            kpss_statistic=np.nan,
            kpss_pvalue=np.nan,
            kpss_rejects=False,
            kpss_pvalue_at_lower_bound=False,
            kpss_pvalue_at_upper_bound=False,
            kpss_ct_pvalue=None,
            kpss_ct_rejects=None,
            acf_decay_lag=-1,
            acf_half_life=-1,
            acf_values=None,
            variance_ratio=np.nan,
            variance_stable=False,
            n_samples=n,
            recommendation=get_recommendation(StationarityType.INSUFFICIENT_DATA),
        )
        if verbose:
            _print_result(r)
        return r

    # --- 1. ADF ---
    adf_stat, adf_p, adf_rejects = compute_adf(y)

    # --- 2. KPSS (level) ---
    kpss_stat, kpss_p, kpss_rejects, kpss_lo, kpss_hi = compute_kpss(y, "c")

    # --- 3. ACF ---
    acf_decay, acf_half, acf_vals = compute_acf_decay(y)

    # --- 4. Variance ratio ---
    var_ratio, var_stable = compute_variance_ratio(y)

    # --- 5. Classify ---
    stype = classify_stationarity(adf_rejects, kpss_rejects)

    # 5b. When both say non-stationary, check for trend-stationarity
    kpss_ct_p: Optional[float] = None
    kpss_ct_rejects: Optional[bool] = None

    if stype == StationarityType.NON_STATIONARY:
        _, ct_p, ct_rejects, _, _ = compute_kpss(y, "ct")
        kpss_ct_p = ct_p
        kpss_ct_rejects = ct_rejects
        if not ct_rejects:
            # KPSS(ct) fails to reject => trend-stationary
            stype = StationarityType.TREND_STATIONARY

    # --- 6. Confidence ---
    conf = assess_confidence(adf_p, kpss_p, kpss_lo, kpss_hi)

    result = StationarityResult(
        stationarity_type=stype,
        is_stationary=(stype == StationarityType.STATIONARY),
        confidence=conf,
        adf_statistic=adf_stat,
        adf_pvalue=adf_p,
        adf_rejects=adf_rejects,
        kpss_statistic=kpss_stat,
        kpss_pvalue=kpss_p,
        kpss_rejects=kpss_rejects,
        kpss_pvalue_at_lower_bound=kpss_lo,
        kpss_pvalue_at_upper_bound=kpss_hi,
        kpss_ct_pvalue=kpss_ct_p,
        kpss_ct_rejects=kpss_ct_rejects,
        acf_decay_lag=acf_decay,
        acf_half_life=acf_half,
        acf_values=acf_vals,
        variance_ratio=var_ratio,
        variance_stable=var_stable,
        n_samples=n,
        recommendation=get_recommendation(stype),
    )

    if verbose:
        _print_result(result)

    return result


# ---------------------------------------------------------------------------
# Benchmark validation
# ---------------------------------------------------------------------------

def validate_level1_benchmarks(verbose: bool = True) -> dict:
    """
    Validate Level 1 against known signals.

    Expected:
        white_noise          -> STATIONARY
        random_walk          -> NON_STATIONARY
        sine_wave            -> STATIONARY  (wide-sense stationary)
        linear_trend_noise   -> TREND_STATIONARY
        ar1_near_unit        -> STATIONARY  (phi=0.9, inside unit circle)
    """
    np.random.seed(42)
    n = 1000

    benchmarks = {
        "white_noise": {
            "signal": np.random.randn(n),
            "expected": StationarityType.STATIONARY,
        },
        "random_walk": {
            "signal": np.cumsum(np.random.randn(n)),
            "expected": StationarityType.NON_STATIONARY,
        },
        "sine_wave": {
            "signal": np.sin(2 * np.pi * 0.05 * np.arange(n)),
            "expected": StationarityType.STATIONARY,
        },
        "linear_trend_noise": {
            "signal": 0.01 * np.arange(n) + np.random.randn(n) * 0.5,
            "expected": StationarityType.TREND_STATIONARY,
        },
        "ar1_near_unit": {
            "signal": _generate_ar1(n, phi=0.9),
            "expected": StationarityType.STATIONARY,
        },
    }

    results = {}
    passed = 0

    if verbose:
        print("=" * 60)
        print("Level 1 Typology — Benchmark Validation")
        print("=" * 60)

    for name, spec in benchmarks.items():
        result = test_stationarity(spec["signal"])
        expected = spec["expected"]
        ok = result.stationarity_type == expected
        if ok:
            passed += 1

        results[name] = {
            "result": result,
            "expected": expected,
            "passed": ok,
        }

        if verbose:
            tag = "PASS" if ok else "FAIL"
            print(f"\n  {name}:")
            print(f"    expected : {expected.value}")
            print(f"    got      : {result.stationarity_type.value}")
            print(f"    ADF p={result.adf_pvalue:.4f}  KPSS p={result.kpss_pvalue:.4f}")
            print(f"    confidence={result.confidence.value}  var_ratio={result.variance_ratio:.2f}")
            print(f"    [{tag}]")

    if verbose:
        print("\n" + "=" * 60)
        print(f"  {passed}/{len(benchmarks)} passed")
        print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_ar1(n: int, phi: float = 0.9) -> np.ndarray:
    """AR(1) process:  y[t] = phi * y[t-1] + eps."""
    y = np.zeros(n)
    eps = np.random.randn(n)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


def _print_result(r: StationarityResult) -> None:
    """Pretty-print a StationarityResult."""
    print(f"=== Level 1 Stationarity (n={r.n_samples}) ===")
    print(f"  ADF  : stat={r.adf_statistic:+.4f}  p={r.adf_pvalue:.4f}  rejects={r.adf_rejects}")

    kpss_note = ""
    if r.kpss_pvalue_at_lower_bound:
        kpss_note = "  (true p < 0.01)"
    elif r.kpss_pvalue_at_upper_bound:
        kpss_note = "  (true p > 0.10)"
    print(f"  KPSS : stat={r.kpss_statistic:.4f}  p={r.kpss_pvalue:.4f}  rejects={r.kpss_rejects}{kpss_note}")

    if r.kpss_ct_pvalue is not None:
        print(f"  KPSS(ct): p={r.kpss_ct_pvalue:.4f}  rejects={r.kpss_ct_rejects}")

    print(f"  ACF  : decay_lag={r.acf_decay_lag}  half_life={r.acf_half_life}")
    print(f"  Var  : ratio={r.variance_ratio:.3f}  stable={r.variance_stable}")
    print(f"  => {r.stationarity_type.value}  (confidence={r.confidence.value})")
    print(f"  => {r.recommendation}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    validate_level1_benchmarks(verbose=True)
