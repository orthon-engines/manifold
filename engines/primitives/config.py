"""
ENGINES Primitives Configuration

Centralized configuration for all primitive function defaults.
Avoids hardcoded magic numbers scattered across modules.

Usage:
    from engines.primitives.config import PRIMITIVES_CONFIG as cfg

    # Access values
    r = cfg.entropy.tolerance_ratio * np.std(signal)
    if n < cfg.min_samples.hurst:
        return np.nan
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EntropyConfig:
    """Configuration for entropy-based measures."""

    # Tolerance ratio for sample/approximate entropy (r = ratio * std)
    tolerance_ratio: float = 0.2

    # Default max scale for multiscale entropy
    max_scale: int = 10

    # Default pattern length (embedding dimension)
    default_m: int = 2

    # Default permutation order
    default_order: int = 3


@dataclass(frozen=True)
class DynamicsConfig:
    """Configuration for dynamical systems analysis."""

    # Lyapunov exponent estimation
    min_separation: int = 10      # Minimum temporal separation for neighbors
    max_steps: int = 10           # Maximum divergence tracking steps

    # Embedding dimension estimation (FNN)
    max_embedding_dim: int = 10   # Maximum dimension to test
    min_vectors_fnn: int = 100    # Minimum embedded vectors for FNN
    fnn_threshold: float = 0.01   # FNN ratio threshold (1%)

    # Recurrence quantification analysis (RQA)
    rqa_threshold_ratio: float = 0.1   # Threshold = ratio * max_distance
    rqa_min_samples: int = 10          # Minimum embedded vectors for RQA

    # Optimal delay estimation
    max_lag_ratio: float = 0.25   # max_lag = ratio * n


@dataclass(frozen=True)
class FractalConfig:
    """Configuration for fractal and long-range dependence analysis."""

    # Rescaled range (R/S) analysis
    rs_min_k: int = 10           # Minimum segment size for R/S
    rs_max_k_ratio: float = 0.5  # max_k = ratio * n
    rs_max_k_cap: int = 100      # Cap on max_k

    # Detrended fluctuation analysis (DFA)
    dfa_min_scale: int = 10      # Minimum scale
    dfa_max_scale_ratio: float = 0.25   # max_scale = ratio * n
    dfa_max_scale_cap: int = 100        # Cap on max_scale
    dfa_n_scales: int = 20              # Number of scale points


@dataclass(frozen=True)
class MinSamplesConfig:
    """Minimum sample requirements for various algorithms."""

    # Fractal/memory analysis
    hurst: int = 20              # Hurst exponent
    dfa: int = 20                # Detrended fluctuation analysis

    # Stationarity tests
    stationarity: int = 20       # ADF, KPSS tests
    changepoints: int = 10       # Changepoint detection
    trend: int = 3               # Linear trend detection

    # Information theory
    transfer_entropy: int = 10   # Transfer entropy
    granger: int = 10            # Granger causality (plus max_lag)

    # Entropy
    sample_entropy: int = 3      # m + 1 minimum

    # Dynamics
    lyapunov: int = 20           # min_separation * 2
    rqa: int = 10                # RQA embedding

    # Correlation
    pacf: int = 22               # max_lag + 2


@dataclass(frozen=True)
class CorrelationConfig:
    """Configuration for correlation analysis."""

    # Partial autocorrelation
    default_max_lag: int = 20


@dataclass(frozen=True)
class InformationConfig:
    """Configuration for information-theoretic measures."""

    # Discretization bins
    default_n_bins: int = 8

    # Transfer entropy / Granger
    default_history_length: int = 1   # k, l parameters
    default_max_lag: int = 5          # Granger max lag


@dataclass(frozen=True)
class ComplexityConfig:
    """Configuration for complexity measures."""

    # Correlation dimension embedding
    default_embed_dim: int = 5

    # Radius scaling for correlation dimension
    radius_min_ratio: float = 0.01   # min_radius = ratio * max_dist
    radius_max_ratio: float = 0.5    # max_radius = ratio * max_dist
    n_radii: int = 10                # Number of radius values


@dataclass(frozen=True)
class PrimitivesConfig:
    """Master configuration for all primitives."""

    entropy: EntropyConfig = EntropyConfig()
    dynamics: DynamicsConfig = DynamicsConfig()
    fractal: FractalConfig = FractalConfig()
    min_samples: MinSamplesConfig = MinSamplesConfig()
    correlation: CorrelationConfig = CorrelationConfig()
    information: InformationConfig = InformationConfig()
    complexity: ComplexityConfig = ComplexityConfig()


# Global singleton instance
PRIMITIVES_CONFIG = PrimitivesConfig()
