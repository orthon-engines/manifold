"""pmtvs import resolver — works with both dev (0.3.x) and published (0.1.4).

Dev pmtvs (editable install, 0.3.x): all 246 functions at top level via sub-packages.
Published pmtvs (PyPI, 0.1.4): 11 at top level, rest via old-style submodule paths.

Usage:
    from manifold.core._pmtvs import eigendecomposition, shannon_entropy
"""

import pmtvs as _pmtvs

# Detect which version we have
_HAS_ALL = hasattr(_pmtvs, "shannon_entropy")

# --- Category B: always at top level (both versions) ---
from pmtvs import (  # noqa: F401, E402
    ftle_direct_perturbation,
    ftle_local_linearization,
    hurst_exponent,
    lyapunov_rosenstein,
    lyapunov_kantz,
    optimal_delay,
    optimal_dimension,
    permutation_entropy,
    sample_entropy,
    time_delay_embedding,
)

if _HAS_ALL:
    # Dev pmtvs — everything at top level
    from pmtvs import (  # noqa: F401
        # information
        shannon_entropy,
        conditional_entropy,
        joint_entropy,
        renyi_entropy,
        mutual_information,
        transfer_entropy,
        kl_divergence,
        js_divergence,
        granger_causality,
        # pairwise
        correlation,
        cross_correlation,
        lag_at_max_xcorr,
        kendall_tau,
        dynamic_time_warping,
        euclidean_distance,
        cosine_similarity,
        linear_regression,
        # matrix
        eigendecomposition,
        covariance_matrix,
        correlation_matrix,
        recurrence_matrix,
        dynamic_mode_decomposition,
        dmd_frequencies,
        dmd_growth_rates,
        # dynamical
        recurrence_rate,
        determinism,
        laminarity,
        trapping_time,
        entropy_rqa,
        correlation_dimension,
        cao_embedding_analysis,
        detect_saddle_points,
        classify_jacobian_eigenvalues,
        compute_separatrix_distance,
        compute_basin_stability,
        compute_variable_sensitivity,
        compute_sensitivity_evolution,
        detect_sensitivity_transitions,
        compute_influence_matrix,
        # topology
        persistence_diagram,
        betti_numbers,
        persistence_entropy,
        # spectral
        dominant_frequency,
        psd,
        spectral_centroid,
        spectral_bandwidth,
        spectral_entropy,
        spectral_flatness,
        hilbert_transform,
        envelope,
        # individual
        autocorrelation,
        dfa,
        approximate_entropy,
        first_derivative,
        second_derivative,
        jerk,
        attractor_reconstruction,
        rate_of_change,
        trend,
        mann_kendall_test,
        changepoints,
        zscore_normalize,
        condition_number,
        effective_dimension,
        # statistics
        kurtosis,
        # stat_tests
        adf_test,
    )
else:
    # Published pmtvs 0.1.4 — use old-style submodule paths
    from pmtvs.information.entropy import shannon_entropy  # noqa: F401
    from pmtvs.information.entropy import conditional_entropy  # noqa: F401
    from pmtvs.information.entropy import joint_entropy  # noqa: F401
    from pmtvs.information.mutual import mutual_information  # noqa: F401
    from pmtvs.information.transfer import transfer_entropy  # noqa: F401
    from pmtvs.information.divergence import kl_divergence  # noqa: F401
    from pmtvs.information.divergence import js_divergence  # noqa: F401
    from pmtvs.pairwise.causality import granger_causality  # noqa: F401
    from pmtvs.pairwise.correlation import correlation  # noqa: F401
    from pmtvs.pairwise.correlation import cross_correlation  # noqa: F401
    from pmtvs.pairwise.correlation import lag_at_max_xcorr  # noqa: F401
    from pmtvs.pairwise.correlation import kendall_tau  # noqa: F401
    from pmtvs.pairwise.distance import dynamic_time_warping  # noqa: F401
    from pmtvs.pairwise.distance import euclidean_distance  # noqa: F401
    from pmtvs.pairwise.distance import cosine_similarity  # noqa: F401
    from pmtvs.pairwise.regression import linear_regression  # noqa: F401
    from pmtvs.matrix.decomposition import eigendecomposition  # noqa: F401
    from pmtvs.matrix.covariance import covariance_matrix  # noqa: F401
    from pmtvs.matrix.covariance import correlation_matrix  # noqa: F401
    from pmtvs.matrix.graph import recurrence_matrix  # noqa: F401
    from pmtvs.matrix.dmd import dynamic_mode_decomposition  # noqa: F401
    from pmtvs.matrix.dmd import dmd_frequencies  # noqa: F401
    from pmtvs.matrix.dmd import dmd_growth_rates  # noqa: F401
    from pmtvs.dynamical.rqa import recurrence_rate  # noqa: F401
    from pmtvs.dynamical.rqa import determinism  # noqa: F401
    from pmtvs.dynamical.rqa import laminarity  # noqa: F401
    from pmtvs.dynamical.rqa import trapping_time  # noqa: F401
    from pmtvs.dynamical.rqa import entropy_rqa  # noqa: F401
    from pmtvs.dynamical.dimension import correlation_dimension  # noqa: F401
    from pmtvs.embedding.delay import cao_embedding_analysis  # noqa: F401
    from pmtvs.dynamical.saddle import detect_saddle_points  # noqa: F401
    from pmtvs.dynamical.saddle import classify_jacobian_eigenvalues  # noqa: F401
    from pmtvs.dynamical.saddle import compute_separatrix_distance  # noqa: F401
    from pmtvs.dynamical.saddle import compute_basin_stability  # noqa: F401
    from pmtvs.dynamical.sensitivity import compute_variable_sensitivity  # noqa: F401
    from pmtvs.dynamical.sensitivity import compute_sensitivity_evolution  # noqa: F401
    from pmtvs.dynamical.sensitivity import detect_sensitivity_transitions  # noqa: F401
    from pmtvs.dynamical.sensitivity import compute_influence_matrix  # noqa: F401
    from pmtvs.topology.persistence import persistence_diagram  # noqa: F401
    from pmtvs.topology.persistence import betti_numbers  # noqa: F401
    from pmtvs.topology.persistence import persistence_entropy  # noqa: F401
    from pmtvs.individual.spectral import dominant_frequency  # noqa: F401
    from pmtvs.individual.spectral import psd  # noqa: F401
    from pmtvs.individual.spectral import spectral_centroid  # noqa: F401
    from pmtvs.individual.spectral import spectral_bandwidth  # noqa: F401
    from pmtvs.individual.spectral import spectral_entropy  # noqa: F401
    from pmtvs.individual.hilbert import hilbert_transform  # noqa: F401
    from pmtvs.individual.hilbert import envelope  # noqa: F401
    from pmtvs.individual.correlation import autocorrelation  # noqa: F401
    from pmtvs.individual.fractal import dfa  # noqa: F401
    from pmtvs.individual.entropy import approximate_entropy  # noqa: F401
    from pmtvs.individual.derivatives import first_derivative  # noqa: F401
    from pmtvs.individual.derivatives import second_derivative  # noqa: F401
    from pmtvs.individual.derivatives import jerk  # noqa: F401
    from pmtvs.individual.dynamics import attractor_reconstruction  # noqa: F401
    from pmtvs.individual.temporal import rate_of_change  # noqa: F401
    from pmtvs.individual.stationarity import trend  # noqa: F401
    from pmtvs.individual.stationarity import mann_kendall_test  # noqa: F401
    from pmtvs.individual.stationarity import changepoints  # noqa: F401
    from pmtvs.individual.normalization import zscore_normalize  # noqa: F401
    from pmtvs.individual.geometry import condition_number  # noqa: F401
    from pmtvs.individual.geometry import effective_dimension  # noqa: F401
    from pmtvs.stat_tests.stationarity_tests import adf_test  # noqa: F401

    # These don't exist in 0.1.4 at all — sentinel to trigger _compat usage
    renyi_entropy = None  # noqa: F811
    spectral_flatness = None  # noqa: F811
    kurtosis = None  # noqa: F811
