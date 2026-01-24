"""
ORTHON Paragraph Engine
========================

Dynamic narrative generation for domain-agnostic signal analysis.

Transforms quantitative analysis outputs into contextual, human-readable
narratives that adapt automatically to the data values. Rather than producing
static reports with blank fields to fill in, the engine generates prose where
the sentences themselves change based on what the metrics reveal.

Architecture:
    RAW SIGNAL → METRICS ENGINE → CLASSIFIER → SEMANTIC LABELS → TEMPLATE ENGINE → FINAL REPORT

Template Syntax:
    - Variable interpolation: {{variable_name|format}}
      Formats: fixed:N (decimals), percent (multiply by 100, add %)

    - Conditional blocks: [?condition]content[/]
      Operators: <, >, <=, >=, ==, !=, &&, ||, !
      Access nested properties: classification.label

Usage:
    >>> from prism.signal_typology.paragraph_engine import ParagraphEngine, generate_signal_report
    >>>
    >>> # Quick report generation
    >>> report = generate_signal_report(metrics, signal_name="Temperature Sensor")
    >>> print(report)
    >>>
    >>> # Advanced usage with custom thresholds
    >>> engine = ParagraphEngine()
    >>> engine.thresholds['correlation']['strong_positive']['min'] = 0.8  # Stricter
    >>> report = engine.generate_report(metrics, signal_name="EEG Channel 1")

Author: Ørthon Project
Version: 1.0.0
"""

import re
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


# =============================================================================
# THRESHOLD CLASSIFICATION SYSTEM
# =============================================================================

# Default thresholds for Signal Typology metrics
# Each metric type defines ranges that map to labels and descriptors
THRESHOLDS: Dict[str, Dict[str, Dict[str, Any]]] = {
    # Correlation (-1 to 1, normalized)
    'correlation': {
        'strong_negative': {
            'max': -0.7,
            'label': 'strong negative',
            'descriptor': 'inversely coupled'
        },
        'moderate_negative': {
            'min': -0.7,
            'max': -0.3,
            'label': 'moderate negative',
            'descriptor': 'weakly inverse'
        },
        'weak': {
            'min': -0.3,
            'max': 0.3,
            'label': 'weak',
            'descriptor': 'largely independent'
        },
        'moderate_positive': {
            'min': 0.3,
            'max': 0.7,
            'label': 'moderate positive',
            'descriptor': 'positively associated'
        },
        'strong_positive': {
            'min': 0.7,
            'label': 'strong positive',
            'descriptor': 'tightly coupled'
        }
    },

    # Stationarity (p-values from ADF/KPSS tests)
    'stationarity': {
        'stationary': {
            'max': 0.05,
            'label': 'stationary',
            'descriptor': 'mean-reverting behavior'
        },
        'borderline': {
            'min': 0.05,
            'max': 0.10,
            'label': 'borderline',
            'descriptor': 'ambiguous stability'
        },
        'nonstationary': {
            'min': 0.10,
            'label': 'non-stationary',
            'descriptor': 'trending or drifting dynamics'
        }
    },

    # Periodicity (spectral peak prominence, 0 to 1)
    'periodicity': {
        'none': {
            'max': 0.1,
            'label': 'aperiodic',
            'descriptor': 'no discernible cyclical pattern'
        },
        'weak': {
            'min': 0.1,
            'max': 0.3,
            'label': 'weakly periodic',
            'descriptor': 'faint cyclical tendencies'
        },
        'moderate': {
            'min': 0.3,
            'max': 0.6,
            'label': 'moderately periodic',
            'descriptor': 'clear recurring patterns'
        },
        'strong': {
            'min': 0.6,
            'label': 'strongly periodic',
            'descriptor': 'dominant cyclical behavior'
        }
    },

    # Noise Level (signal-to-noise ratio in dB)
    'noise': {
        'very_noisy': {
            'max': 0,
            'label': 'very noisy',
            'descriptor': 'signal dominated by noise'
        },
        'noisy': {
            'min': 0,
            'max': 10,
            'label': 'noisy',
            'descriptor': 'significant measurement uncertainty'
        },
        'moderate': {
            'min': 10,
            'max': 20,
            'label': 'moderately noisy',
            'descriptor': 'acceptable signal quality'
        },
        'clean': {
            'min': 20,
            'label': 'clean',
            'descriptor': 'high signal fidelity'
        }
    },

    # Complexity (entropy, normalized 0 to 1)
    'complexity': {
        'simple': {
            'max': 0.3,
            'label': 'low complexity',
            'descriptor': 'predictable, ordered dynamics'
        },
        'moderate': {
            'min': 0.3,
            'max': 0.7,
            'label': 'moderate complexity',
            'descriptor': 'mixed deterministic and stochastic elements'
        },
        'complex': {
            'min': 0.7,
            'label': 'high complexity',
            'descriptor': 'rich, irregular behavior patterns'
        }
    },

    # Autocorrelation persistence (lag-1 ACF, 0 to 1)
    'acf_persistence': {
        'low': {
            'max': 0.3,
            'label': 'weak',
            'descriptor': 'rapidly decorrelating'
        },
        'moderate': {
            'min': 0.3,
            'max': 0.7,
            'label': 'moderate',
            'descriptor': 'moderate temporal dependence'
        },
        'high': {
            'min': 0.7,
            'label': 'strong positive',
            'descriptor': 'tightly coupled'
        }
    },

    # Distribution shape (kurtosis)
    'kurtosis': {
        'light_tails': {
            'max': 2.5,
            'label': 'platykurtic',
            'descriptor': 'light tails, bounded dynamics'
        },
        'normal': {
            'min': 2.5,
            'max': 3.5,
            'label': 'mesokurtic',
            'descriptor': 'near-normal tail behavior'
        },
        'heavy_tails': {
            'min': 3.5,
            'label': 'leptokurtic',
            'descriptor': 'heavy tails, elevated probability of extreme values'
        }
    },

    # Skewness
    'skewness': {
        'left_skewed': {
            'max': -0.5,
            'label': 'negatively skewed',
            'descriptor': 'left-tail asymmetry with occasional large negative excursions'
        },
        'symmetric': {
            'min': -0.5,
            'max': 0.5,
            'label': 'symmetric',
            'descriptor': 'near-symmetric distribution'
        },
        'right_skewed': {
            'min': 0.5,
            'label': 'positively skewed',
            'descriptor': 'right-tail asymmetry with occasional large positive excursions'
        }
    }
}

# Signal type classifications based on combined characteristics
SIGNAL_TYPES: Dict[str, Dict[str, str]] = {
    'deterministic_periodic': {
        'label': 'Deterministic Periodic Process',
        'implications': 'harmonic analysis and Fourier-based methods are appropriate'
    },
    'stochastic_stationary': {
        'label': 'Stochastic Stationary Process',
        'implications': 'standard time series methods apply; ARIMA, exponential smoothing, and spectral analysis are appropriate'
    },
    'nonstationary_trending': {
        'label': 'Non-Stationary Trending Process',
        'implications': 'differencing or detrending required before standard time series methods apply'
    },
    'chaotic_complex': {
        'label': 'Chaotic/Complex Process',
        'implications': 'sensitivity to initial conditions; may require nonlinear dynamics approaches'
    },
    'noise_dominated': {
        'label': 'Noise-Dominated Process',
        'implications': 'filtering and noise reduction recommended before further analysis'
    },
    'mixed_regime': {
        'label': 'Mixed-Regime Process',
        'implications': 'regime-switching models or piecewise analysis may be appropriate'
    }
}


@dataclass
class Classification:
    """Result of classifying a numeric value."""
    category: str
    label: str
    descriptor: str
    value: float
    metric_type: str


def classify_value(
    value: float,
    metric_type: str,
    thresholds: Optional[Dict] = None
) -> Classification:
    """
    Map a numeric value to a classification using thresholds.

    Args:
        value: The numeric value to classify
        metric_type: Type of metric (e.g., 'correlation', 'stationarity')
        thresholds: Optional custom thresholds dict

    Returns:
        Classification object with category, label, and descriptor
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    if metric_type not in thresholds:
        return Classification(
            category='unknown',
            label='unknown',
            descriptor='unclassified metric',
            value=value,
            metric_type=metric_type
        )

    metric_thresholds = thresholds[metric_type]

    # Handle NaN values
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return Classification(
            category='insufficient_data',
            label='insufficient data',
            descriptor='data not available',
            value=value,
            metric_type=metric_type
        )

    for category, bounds in metric_thresholds.items():
        min_val = bounds.get('min', float('-inf'))
        max_val = bounds.get('max', float('inf'))

        if min_val <= value < max_val:
            return Classification(
                category=category,
                label=bounds['label'],
                descriptor=bounds['descriptor'],
                value=value,
                metric_type=metric_type
            )

    # Fallback for edge cases (value exactly at boundary)
    last_category = list(metric_thresholds.keys())[-1]
    last_bounds = metric_thresholds[last_category]
    return Classification(
        category=last_category,
        label=last_bounds['label'],
        descriptor=last_bounds['descriptor'],
        value=value,
        metric_type=metric_type
    )


# =============================================================================
# TEMPLATE RENDERING ENGINE
# =============================================================================

class TemplateEngine:
    """
    Renders dynamic paragraph templates with variable interpolation and conditionals.

    Template Syntax:
        - {{variable|format}}: Insert variable with optional format
          - {{value|fixed:3}}: Fixed decimal places
          - {{value|percent}}: Multiply by 100, add %
          - {{value}}: String conversion

        - [?condition]content[/]: Conditional content
          - [?snr > 20]high quality[/]
          - [?adf_pvalue < 0.05 && kpss_pvalue > 0.05]agreement[/]
          - [?!heteroscedastic]stable variance[/]
    """

    # Regex patterns for parsing
    CONDITIONAL_PATTERN = re.compile(
        r'\[\?([^\]]+)\]([\s\S]*?)\[/\]',
        re.MULTILINE
    )
    VARIABLE_PATTERN = re.compile(
        r'\{\{([^}|]+)(?:\|([^}]+))?\}\}'
    )

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """
        Initialize template engine with context.

        Args:
            context: Dictionary of variables available for interpolation
        """
        self.context = context or {}

    def render(self, template: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Render a template with the given context.

        Args:
            template: Template string with {{variables}} and [?conditions]
            context: Optional context to merge with default context

        Returns:
            Rendered string with all substitutions made
        """
        ctx = {**self.context, **(context or {})}

        # Pass 1: Evaluate conditionals
        result = self._evaluate_conditionals(template, ctx)

        # Pass 2: Interpolate variables
        result = self._interpolate_variables(result, ctx)

        # Clean up whitespace
        result = self._clean_whitespace(result)

        return result

    def _evaluate_conditionals(self, template: str, context: Dict[str, Any]) -> str:
        """Evaluate all conditional blocks in template."""
        def replacer(match):
            condition = match.group(1).strip()
            content = match.group(2)

            if self._evaluate_condition(condition, context):
                return content
            return ''

        # Recursively process nested conditionals
        previous = None
        result = template
        while result != previous:
            previous = result
            result = self.CONDITIONAL_PATTERN.sub(replacer, result)

        return result

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition expression.

        Supports: <, >, <=, >=, ==, !=, &&, ||, !
        """
        # Handle logical NOT
        if condition.startswith('!'):
            inner = condition[1:].strip()
            return not self._evaluate_condition(inner, context)

        # Handle logical AND
        if '&&' in condition:
            parts = condition.split('&&')
            return all(self._evaluate_condition(p.strip(), context) for p in parts)

        # Handle logical OR
        if '||' in condition:
            parts = condition.split('||')
            return any(self._evaluate_condition(p.strip(), context) for p in parts)

        # Handle comparison operators
        for op in ['<=', '>=', '!=', '==', '<', '>']:
            if op in condition:
                left, right = condition.split(op, 1)
                left_val = self._get_value(left.strip(), context)
                right_val = self._parse_literal(right.strip(), context)

                if left_val is None or right_val is None:
                    return False

                try:
                    if op == '<':
                        return float(left_val) < float(right_val)
                    elif op == '>':
                        return float(left_val) > float(right_val)
                    elif op == '<=':
                        return float(left_val) <= float(right_val)
                    elif op == '>=':
                        return float(left_val) >= float(right_val)
                    elif op == '==':
                        return str(left_val) == str(right_val)
                    elif op == '!=':
                        return str(left_val) != str(right_val)
                except (ValueError, TypeError):
                    return str(left_val) == str(right_val) if op == '==' else False

        # Boolean check (just a variable name)
        val = self._get_value(condition, context)
        return bool(val)

    def _get_value(self, path: str, context: Dict[str, Any]) -> Any:
        """Get a value from context, supporting dot notation for nested access."""
        parts = path.split('.')
        current = context

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            elif hasattr(current, '__getitem__'):
                try:
                    current = current[part]
                except (KeyError, IndexError, TypeError):
                    return None
            else:
                return None

            if current is None:
                return None

        return current

    def _parse_literal(self, value: str, context: Dict[str, Any]) -> Any:
        """Parse a literal value or look up in context."""
        value = value.strip()

        # String literal
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]

        # Numeric literal
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Boolean literals
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False

        # Context lookup
        return self._get_value(value, context)

    def _interpolate_variables(self, template: str, context: Dict[str, Any]) -> str:
        """Replace all {{variable|format}} patterns with values."""
        def replacer(match):
            var_path = match.group(1).strip()
            format_spec = match.group(2)

            value = self._get_value(var_path, context)

            if value is None:
                return f'[missing: {var_path}]'

            return self._format_value(value, format_spec)

        return self.VARIABLE_PATTERN.sub(replacer, template)

    def _format_value(self, value: Any, format_spec: Optional[str]) -> str:
        """Format a value according to the format specifier."""
        if format_spec is None:
            if isinstance(value, float):
                return f'{value:.4g}'
            return str(value)

        format_spec = format_spec.strip()

        # fixed:N - fixed decimal places
        if format_spec.startswith('fixed:'):
            try:
                decimals = int(format_spec.split(':')[1])
                return f'{float(value):.{decimals}f}'
            except (ValueError, IndexError):
                return str(value)

        # percent - multiply by 100, add %
        if format_spec == 'percent':
            try:
                return f'{float(value) * 100:.1f}%'
            except (ValueError, TypeError):
                return str(value)

        return str(value)

    def _clean_whitespace(self, text: str) -> str:
        """Clean up excess whitespace from rendered text."""
        # Remove blank lines
        lines = [line for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])\s*([A-Z])', r'\1 \2', text)

        return text.strip()


# =============================================================================
# SIGNAL TYPOLOGY TEMPLATES
# =============================================================================

TEMPLATES: Dict[str, Dict[str, str]] = {
    'autocorrelation': {
        'summary': """The {{signal_name}} signal exhibits {{acf_classification.label}} autocorrelation (lag-1 ACF = {{acf_lag1|fixed:3}}), indicating {{acf_classification.descriptor}}. [?acf_lag1 > 0.8]This high persistence suggests the signal carries substantial memory of its recent history, making short-term predictions more tractable but long-term forecasts vulnerable to cumulative drift.[/][?acf_lag1 < 0.2]The rapid decay of autocorrelation indicates that observations become essentially independent over short timescales, characteristic of noise-dominated or highly stochastic processes.[/]""",

        'detail': """Autocorrelation analysis reveals that {{signal_name}} maintains {{acf_interpretation}} across the first {{acf_significant_lags}} lags before decaying below the significance threshold (α = 0.05). The partial autocorrelation function (PACF) shows {{pacf_pattern}}, suggesting [?pacf_cutoff_lag <= 2]an autoregressive process of low order (AR({{pacf_cutoff_lag}})).[/][?pacf_cutoff_lag > 2]a more complex temporal dependency structure potentially requiring higher-order modeling.[/]"""
    },

    'stationarity': {
        'summary': """Stationarity testing indicates that {{signal_name}} is {{stationarity_classification.label}} (ADF p-value = {{adf_pvalue|fixed:4}}, KPSS p-value = {{kpss_pvalue|fixed:4}}). [?adf_pvalue < 0.05 && kpss_pvalue > 0.05]Both tests agree on stationarity, providing high confidence in mean-reverting behavior.[/][?adf_pvalue > 0.05 && kpss_pvalue < 0.05]Both tests agree on non-stationarity, suggesting persistent trends or structural changes.[/][?adf_pvalue < 0.05 && kpss_pvalue < 0.05]The tests yield conflicting results, indicating the signal may be trend-stationary or exhibit regime-dependent behavior.[/][?adf_pvalue > 0.05 && kpss_pvalue > 0.05]The tests yield conflicting results, suggesting the signal may require further investigation.[/]""",

        'detail': """The Augmented Dickey-Fuller test [?adf_pvalue < 0.05]rejects the null hypothesis of a unit root[/][?adf_pvalue >= 0.05]fails to reject the unit root hypothesis[/] at the 5% significance level (test statistic = {{adf_stat|fixed:3}}, critical value = {{adf_critical|fixed:3}}). The KPSS test, which reverses the null hypothesis, [?kpss_pvalue > 0.05]supports stationarity[/][?kpss_pvalue <= 0.05]indicates non-stationarity[/]. [?differencing_required]First-order differencing achieves stationarity with d = {{differencing_order}}.[/]"""
    },

    'spectral': {
        'summary': """Spectral analysis classifies {{signal_name}} as {{periodicity_classification.label}}, with {{periodicity_classification.descriptor}}. [?dominant_period > 0]The dominant period is approximately {{dominant_period|fixed:1}} time units (frequency = {{dominant_freq|fixed:4}} cycles/unit), explaining {{spectral_power_ratio|percent}} of total spectral power.[/][?dominant_period == 0]No dominant frequency emerges above the noise floor.[/]""",

        'detail': """The power spectral density (PSD) reveals [?num_significant_peaks == 0]a flat spectrum consistent with white noise or overdifferenced data[/][?num_significant_peaks == 1]a single dominant peak at frequency {{dominant_freq|fixed:4}}[/][?num_significant_peaks > 1]{{num_significant_peaks}} significant peaks at frequencies {{peak_frequencies}}, suggesting multi-scale periodic behavior or potential harmonic relationships[/]. The spectral entropy of {{spectral_entropy|fixed:3}} indicates {{complexity_classification.descriptor}}. [?spectral_slope < -1]The negative spectral slope (β = {{spectral_slope|fixed:2}}) suggests long-range dependence characteristic of 1/f processes.[/]"""
    },

    'distribution': {
        'summary': """The {{signal_name}} distribution [?normality_pvalue > 0.05]cannot be distinguished from normal at the 5% level (Shapiro-Wilk p = {{normality_pvalue|fixed:3}})[/][?normality_pvalue <= 0.05]departs significantly from normality (Shapiro-Wilk p = {{normality_pvalue|fixed:3}})[/], with skewness = {{skewness|fixed:3}} and kurtosis = {{kurtosis|fixed:3}}. [?kurtosis > 3]Heavy tails indicate elevated probability of extreme values.[/][?kurtosis < 3]Light tails suggest bounded dynamics with limited extreme events.[/]""",

        'detail': """Distributional analysis shows {{signal_name}} has mean = {{mean|fixed:4}}, standard deviation = {{std|fixed:4}}, and coefficient of variation = {{cv|percent}}. [?skewness > 0.5]Positive skewness indicates right-tail asymmetry with occasional large positive excursions.[/][?skewness < -0.5]Negative skewness indicates left-tail asymmetry with occasional large negative excursions.[/][?skewness >= -0.5 && skewness <= 0.5]Near-symmetric distribution.[/] The interquartile range (IQR = {{iqr|fixed:4}}) spans {{iqr_range_description}}."""
    },

    'noise': {
        'summary': """Signal quality assessment indicates {{noise_classification.label}} conditions (SNR = {{snr|fixed:1}} dB), characterized by {{noise_classification.descriptor}}. [?snr > 20]Downstream analyses can proceed with high confidence in the underlying signal.[/][?snr <= 10]Noise mitigation techniques such as smoothing or filtering should be considered before further analysis.[/]""",

        'detail': """Noise decomposition separates {{signal_name}} into trend, seasonal, and residual components. The residual exhibits [?residual_autocorr < 0.1]white noise characteristics (residual ACF ≈ 0), validating the decomposition[/][?residual_autocorr >= 0.1]significant autocorrelation (residual ACF = {{residual_autocorr|fixed:3}}), suggesting unmodeled dynamics remain[/]. [?heteroscedastic]Variance is non-constant over time (heteroscedastic), potentially requiring variance-stabilizing transformations.[/][?!heteroscedastic]Variance remains stable across the observation window (homoscedastic).[/]"""
    },

    'integrated': {
        'summary': """Based on the above characterization, {{signal_name}} is classified as: {{signal_type_label}}. This classification implies {{signal_type_implications}}.""",

        'detail': """The integrated signal typology combines stationarity ({{stationarity_classification.label}}), periodicity ({{periodicity_classification.label}}), noise level ({{noise_classification.label}}), and complexity ({{complexity_classification.label}}) into a composite characterization. [?signal_type == "deterministic_periodic"]The signal follows predictable cyclical patterns amenable to harmonic analysis and Fourier-based methods.[/][?signal_type == "stochastic_stationary"]The signal represents a stationary random process suitable for ARIMA modeling and standard statistical inference.[/][?signal_type == "nonstationary_trending"]The signal exhibits persistent trends requiring differencing or detrending before modeling.[/][?signal_type == "chaotic_complex"]The signal displays sensitivity to initial conditions and may require nonlinear dynamics approaches.[/][?signal_type == "noise_dominated"]The signal is dominated by noise; filtering recommended before analysis.[/][?signal_type == "mixed_regime"]The signal exhibits multiple behavioral regimes; regime-switching models may be appropriate.[/]"""
    }
}


# =============================================================================
# PARAGRAPH ENGINE
# =============================================================================

class ParagraphEngine:
    """
    Main engine for generating dynamic narrative reports from signal metrics.

    Combines threshold classification with template rendering to produce
    human-readable narratives that adapt to the data.
    """

    def __init__(
        self,
        thresholds: Optional[Dict] = None,
        templates: Optional[Dict] = None
    ):
        """
        Initialize the Paragraph Engine.

        Args:
            thresholds: Custom threshold definitions (merges with defaults)
            templates: Custom templates (merges with defaults)
        """
        self.thresholds = {**THRESHOLDS, **(thresholds or {})}
        self.templates = {**TEMPLATES, **(templates or {})}
        self.template_engine = TemplateEngine()

    def classify_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Classification]:
        """
        Classify all relevant metrics in a metrics dict.

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            Dictionary of metric -> Classification
        """
        classifications = {}

        # Map metric names to classification types
        metric_mapping = {
            'acf_lag1': 'acf_persistence',
            'adf_pvalue': 'stationarity',
            'kpss_pvalue': 'stationarity',
            'spectral_power_ratio': 'periodicity',
            'snr': 'noise',
            'spectral_entropy': 'complexity',
            'kurtosis': 'kurtosis',
            'skewness': 'skewness',
        }

        for metric_name, classification_type in metric_mapping.items():
            if metric_name in metrics and metrics[metric_name] is not None:
                classifications[f'{metric_name}_classification'] = classify_value(
                    metrics[metric_name],
                    classification_type,
                    self.thresholds
                )

        # Create derived classifications
        classifications['acf_classification'] = classifications.get(
            'acf_lag1_classification',
            Classification('unknown', 'unknown', 'unclassified', 0.0, 'acf')
        )

        classifications['stationarity_classification'] = self._determine_stationarity(metrics)
        classifications['periodicity_classification'] = classifications.get(
            'spectral_power_ratio_classification',
            classify_value(metrics.get('spectral_power_ratio', 0), 'periodicity', self.thresholds)
        )
        classifications['noise_classification'] = classifications.get(
            'snr_classification',
            classify_value(metrics.get('snr', 10), 'noise', self.thresholds)
        )
        classifications['complexity_classification'] = self._determine_complexity(metrics)

        return classifications

    def _determine_stationarity(self, metrics: Dict[str, Any]) -> Classification:
        """Determine overall stationarity classification from test results."""
        adf_p = metrics.get('adf_pvalue', 1.0)
        kpss_p = metrics.get('kpss_pvalue', 0.0)

        # ADF: reject null (has unit root) if p < 0.05 -> stationary
        # KPSS: reject null (is stationary) if p < 0.05 -> non-stationary
        adf_says_stationary = adf_p < 0.05
        kpss_says_stationary = kpss_p > 0.05

        if adf_says_stationary and kpss_says_stationary:
            return Classification(
                'stationary', 'stationary', 'mean-reverting behavior',
                adf_p, 'stationarity'
            )
        elif not adf_says_stationary and not kpss_says_stationary:
            return Classification(
                'nonstationary', 'non-stationary', 'trending or drifting dynamics',
                adf_p, 'stationarity'
            )
        else:
            return Classification(
                'borderline', 'borderline', 'ambiguous stability',
                adf_p, 'stationarity'
            )

    def _determine_complexity(self, metrics: Dict[str, Any]) -> Classification:
        """Determine complexity classification from entropy measures."""
        # Use spectral entropy if available, otherwise sample/permutation entropy
        entropy = metrics.get('spectral_entropy')
        if entropy is None:
            entropy = metrics.get('sample_entropy', metrics.get('permutation_entropy', 0.5))

        # Normalize entropy to 0-1 range
        if entropy is not None:
            # Sample entropy typically ranges 0-3+, normalize
            if entropy > 1:
                entropy = min(entropy / 3.0, 1.0)

        return classify_value(entropy or 0.5, 'complexity', self.thresholds)

    def determine_signal_type(
        self,
        metrics: Dict[str, Any],
        classifications: Dict[str, Classification]
    ) -> Tuple[str, str, str]:
        """
        Determine overall signal type from classifications.

        Args:
            metrics: Raw metrics dict
            classifications: Classification objects

        Returns:
            Tuple of (signal_type_code, label, implications)
        """
        stationarity = classifications.get('stationarity_classification')
        periodicity = classifications.get('periodicity_classification')
        noise = classifications.get('noise_classification')
        complexity = classifications.get('complexity_classification')

        # Decision logic for signal type
        is_stationary = stationarity and stationarity.category == 'stationary'
        is_periodic = periodicity and periodicity.category in ['moderate', 'strong']
        is_noisy = noise and noise.category in ['noisy', 'very_noisy']
        is_complex = complexity and complexity.category == 'complex'

        if is_noisy and metrics.get('snr', 10) < 5:
            signal_type = 'noise_dominated'
        elif is_periodic and is_stationary:
            signal_type = 'deterministic_periodic'
        elif is_stationary and not is_complex:
            signal_type = 'stochastic_stationary'
        elif not is_stationary:
            signal_type = 'nonstationary_trending'
        elif is_complex:
            signal_type = 'chaotic_complex'
        else:
            signal_type = 'mixed_regime'

        type_info = SIGNAL_TYPES.get(signal_type, SIGNAL_TYPES['mixed_regime'])
        return signal_type, type_info['label'], type_info['implications']

    def build_context(
        self,
        metrics: Dict[str, Any],
        signal_name: str = "Signal"
    ) -> Dict[str, Any]:
        """
        Build complete context for template rendering.

        Args:
            metrics: Raw metrics dict
            signal_name: Human-readable signal name

        Returns:
            Context dict with all values and classifications
        """
        # Start with raw metrics
        context = dict(metrics)
        context['signal_name'] = signal_name

        # Add classifications
        classifications = self.classify_metrics(metrics)
        for name, classification in classifications.items():
            context[name] = {
                'category': classification.category,
                'label': classification.label,
                'descriptor': classification.descriptor
            }

        # Add signal type
        signal_type, label, implications = self.determine_signal_type(
            metrics, classifications
        )
        context['signal_type'] = signal_type
        context['signal_type_label'] = label
        context['signal_type_implications'] = implications

        # Add derived values with defaults
        context.setdefault('acf_interpretation', 'moderate persistence')
        context.setdefault('acf_significant_lags', 5)
        context.setdefault('pacf_pattern', 'exponential decay')
        context.setdefault('pacf_cutoff_lag', 2)
        context.setdefault('differencing_required', False)
        context.setdefault('differencing_order', 0)
        context.setdefault('dominant_period', 0)
        context.setdefault('dominant_freq', 0)
        context.setdefault('spectral_power_ratio', 0)
        context.setdefault('num_significant_peaks', 0)
        context.setdefault('peak_frequencies', '')
        context.setdefault('spectral_slope', 0)
        context.setdefault('iqr_range_description', 'the middle 50% of observations')
        context.setdefault('residual_autocorr', 0)
        context.setdefault('heteroscedastic', False)

        # Calculate CV if not present
        if 'cv' not in context and 'std' in context and 'mean' in context:
            mean = context['mean']
            if mean and mean != 0:
                context['cv'] = context['std'] / abs(mean)
            else:
                context['cv'] = 0

        return context

    def render_section(
        self,
        section: str,
        context: Dict[str, Any],
        detail_level: str = 'summary'
    ) -> str:
        """
        Render a single section of the report.

        Args:
            section: Section name (e.g., 'autocorrelation', 'stationarity')
            context: Rendering context
            detail_level: 'summary' or 'detail'

        Returns:
            Rendered section text
        """
        if section not in self.templates:
            return f"[Unknown section: {section}]"

        template = self.templates[section].get(detail_level, '')
        return self.template_engine.render(template, context)

    def generate_report(
        self,
        metrics: Dict[str, Any],
        signal_name: str = "Signal",
        sections: Optional[List[str]] = None,
        detail_level: str = 'summary'
    ) -> str:
        """
        Generate a complete signal typology report.

        Args:
            metrics: Dictionary of computed metrics
            signal_name: Human-readable signal name
            sections: List of sections to include (None = all)
            detail_level: 'summary' or 'detail'

        Returns:
            Complete rendered report
        """
        context = self.build_context(metrics, signal_name)

        if sections is None:
            sections = ['autocorrelation', 'stationarity', 'spectral',
                       'distribution', 'noise', 'integrated']

        paragraphs = []
        for section in sections:
            rendered = self.render_section(section, context, detail_level)
            if rendered.strip():
                paragraphs.append(rendered)

        return '\n\n'.join(paragraphs)

    def generate_comparison_report(
        self,
        signals: Dict[str, Dict[str, Any]],
        sections: Optional[List[str]] = None
    ) -> str:
        """
        Generate a comparison report for multiple signals.

        Args:
            signals: Dict of signal_name -> metrics
            sections: Sections to compare

        Returns:
            Comparison report text
        """
        if sections is None:
            sections = ['stationarity', 'periodicity', 'noise', 'integrated']

        contexts = {
            name: self.build_context(metrics, name)
            for name, metrics in signals.items()
        }

        lines = ["# Signal Comparison Report\n"]

        for section in sections:
            lines.append(f"\n## {section.title()}\n")

            for name, ctx in contexts.items():
                rendered = self.render_section(section, ctx, 'summary')
                if rendered.strip():
                    lines.append(rendered)
                    lines.append("")

        return '\n'.join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_signal_report(
    metrics: Dict[str, Any],
    signal_name: str = "Signal",
    detail_level: str = 'summary'
) -> str:
    """
    Quick function to generate a signal typology report.

    Args:
        metrics: Dictionary of computed metrics
        signal_name: Human-readable signal name
        detail_level: 'summary' or 'detail'

    Returns:
        Complete rendered report
    """
    engine = ParagraphEngine()
    return engine.generate_report(metrics, signal_name, detail_level=detail_level)


def classify_signal(
    metrics: Dict[str, Any],
    signal_name: str = "Signal"
) -> Dict[str, Any]:
    """
    Classify a signal and return structured results.

    Args:
        metrics: Dictionary of computed metrics
        signal_name: Human-readable signal name

    Returns:
        Dictionary with classifications and signal type
    """
    engine = ParagraphEngine()
    classifications = engine.classify_metrics(metrics)
    signal_type, label, implications = engine.determine_signal_type(
        metrics, classifications
    )

    return {
        'signal_name': signal_name,
        'classifications': {
            name: {
                'category': c.category,
                'label': c.label,
                'descriptor': c.descriptor,
                'value': c.value
            }
            for name, c in classifications.items()
        },
        'signal_type': signal_type,
        'signal_type_label': label,
        'signal_type_implications': implications
    }


def render_template(template: str, context: Dict[str, Any]) -> str:
    """
    Render a custom template with the given context.

    Args:
        template: Template string
        context: Variable context

    Returns:
        Rendered string
    """
    engine = TemplateEngine(context)
    return engine.render(template)


# =============================================================================
# INTEGRATION WITH PRISM SIGNAL TYPOLOGY
# =============================================================================

def metrics_to_paragraph_context(
    prism_metrics: Dict[str, Any],
    prism_profile: Dict[str, float],
    signal_name: str = "Signal"
) -> Dict[str, Any]:
    """
    Convert PRISM Signal Typology metrics to Paragraph Engine context.

    This bridges the PRISM 9-axis system with the Paragraph Engine's
    metric expectations.

    Args:
        prism_metrics: Raw metrics from orchestrator.compute_metrics()
        prism_profile: Normalized profile from normalize.metrics_to_profile()
        signal_name: Human-readable signal name

    Returns:
        Context dict suitable for Paragraph Engine
    """
    context = {
        'signal_name': signal_name,
        'n_samples': prism_metrics.get('n_samples', 0),

        # Autocorrelation (from memory axis)
        'acf_lag1': prism_metrics.get('acf_lag1', 0.5),
        'hurst_exponent': prism_metrics.get('hurst_exponent', 0.5),
        'acf_decay_rate': prism_metrics.get('acf_decay_rate', 0),

        # We don't have ADF/KPSS in PRISM by default, so derive from Hurst
        # H < 0.5: mean-reverting (stationary)
        # H > 0.5: trending (non-stationary)
        'adf_pvalue': 0.01 if prism_metrics.get('hurst_exponent', 0.5) < 0.5 else 0.2,
        'kpss_pvalue': 0.2 if prism_metrics.get('hurst_exponent', 0.5) < 0.5 else 0.01,

        # Spectral (from frequency axis)
        'spectral_entropy': prism_metrics.get('spectral_entropy', 0.5),
        'dominant_frequency': prism_metrics.get('dominant_frequency', 0),
        'dominant_freq': prism_metrics.get('dominant_frequency', 0),
        'dominant_period': 1.0 / prism_metrics.get('dominant_frequency', 0.001)
            if prism_metrics.get('dominant_frequency', 0) > 0 else 0,
        'spectral_power_ratio': 1.0 - prism_metrics.get('spectral_entropy', 0.5),

        # Distribution (from derivatives axis)
        'derivative_kurtosis': prism_metrics.get('derivative_kurtosis', 3.0),
        'kurtosis': prism_metrics.get('derivative_kurtosis', 3.0),
        'skewness': 0,  # Not in PRISM metrics by default

        # Noise (derive from signal quality)
        # Use inverse of entropy as proxy for SNR
        'snr': 20 * (1.0 - prism_metrics.get('spectral_entropy', 0.5)) + 5,

        # Entropy measures
        'sample_entropy': prism_metrics.get('sample_entropy', 1.0),
        'permutation_entropy': prism_metrics.get('permutation_entropy', 0.5),

        # Volatility
        'garch_persistence': prism_metrics.get('garch_persistence', 0.5),
        'heteroscedastic': prism_metrics.get('garch_persistence', 0.5) > 0.7,

        # Recurrence
        'recurrence_rate': prism_metrics.get('recurrence_rate', 0),
        'determinism': prism_metrics.get('determinism', 0),

        # Wavelet
        'wavelet_entropy': prism_metrics.get('wavelet_entropy', 0.5),

        # Include normalized profile scores
        'memory_score': prism_profile.get('memory', 0.5),
        'information_score': prism_profile.get('information', 0.5),
        'frequency_score': prism_profile.get('frequency', 0.5),
        'volatility_score': prism_profile.get('volatility', 0.5),
        'wavelet_score': prism_profile.get('wavelet', 0.5),
        'derivatives_score': prism_profile.get('derivatives', 0.5),
        'recurrence_score': prism_profile.get('recurrence', 0.5),
        'discontinuity_score': prism_profile.get('discontinuity', 0.5),
        'momentum_score': prism_profile.get('momentum', 0.5),
    }

    return context


def generate_prism_report(
    prism_metrics: Dict[str, Any],
    prism_profile: Dict[str, float],
    signal_name: str = "Signal",
    detail_level: str = 'summary'
) -> str:
    """
    Generate a Paragraph Engine report from PRISM Signal Typology output.

    Args:
        prism_metrics: Raw metrics from orchestrator.compute_metrics()
        prism_profile: Normalized profile from normalize.metrics_to_profile()
        signal_name: Human-readable signal name
        detail_level: 'summary' or 'detail'

    Returns:
        Complete rendered report
    """
    context = metrics_to_paragraph_context(prism_metrics, prism_profile, signal_name)
    engine = ParagraphEngine()
    return engine.generate_report(context, signal_name, detail_level=detail_level)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    'ParagraphEngine',
    'TemplateEngine',
    'Classification',

    # Classification
    'classify_value',
    'classify_signal',
    'THRESHOLDS',
    'SIGNAL_TYPES',

    # Templates
    'TEMPLATES',
    'render_template',

    # Report generation
    'generate_signal_report',

    # PRISM integration
    'metrics_to_paragraph_context',
    'generate_prism_report',
]
