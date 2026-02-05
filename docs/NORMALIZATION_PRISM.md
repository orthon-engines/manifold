# Normalization in PRISM/ORTHON

**Version:** 2.5
**Last Updated:** 2026-02-05

## Overview

Normalization is a critical preprocessing step that makes features comparable across different scales. However, different normalization methods have different trade-offs, particularly regarding sensitivity to outliers and distributional assumptions.

```
PRISM computes normalized values. ORTHON interprets what they mean.
```

---

## Quick Reference

| Method | Formula | Robustness | Best For |
|--------|---------|------------|----------|
| **Z-Score** | (x - μ) / σ | Low | Clean Gaussian data |
| **Robust** | (x - median) / IQR | Medium | Moderate outliers |
| **MAD** | (x - median) / MAD | High | Heavy tails, industrial data |
| **Min-Max** | (x - min) / (max - min) | Very Low | Bounded output required |
| **None** | x | N/A | Preserve raw variance dynamics |

---

## Methods in Detail

### 1. Z-Score (Standard Normalization)

```python
z = (x - mean) / std
```

**Advantages:**
- Most efficient for truly Gaussian data
- Well-understood statistical properties
- Transforms Gaussian to N(0,1)

**Risks:**
- **Outlier sensitivity**: A single extreme value inflates std, compressing the entire distribution
- **Gaussian assumption**: Misrepresents heavy-tailed distributions
- **Masking effect**: Outliers inflate std, making subsequent outliers appear less extreme
- **Temporal contamination**: Global mean/std means future anomalies affect past normalization

**Use when:**
- Data is approximately Gaussian
- No significant outliers present
- Baseline period is known to be clean

**Example - Outlier Problem:**
```
Data: [1, 2, 3, 4, 5, 1000]  # Outlier at 1000
Mean: 169.2, Std: 399.8
Z-score of 1000: 2.08  # Looks "only" 2 std away!
Z-score of 5: -0.41    # Normal values compressed to near-zero
```

---

### 2. Robust Scaling (IQR-based)

```python
robust = (x - median) / IQR
IQR = Q3 - Q1  # Interquartile range (75th - 25th percentile)
```

**Advantages:**
- Median has 50% breakdown point (tolerates up to 50% outliers)
- IQR ignores extreme tails
- Better for skewed distributions

**Risks:**
- Less efficient than std for Gaussian data
- Assumes roughly symmetric distribution around median
- Can be unstable for very small datasets

**Use when:**
- Outliers are present or suspected
- Distribution is roughly symmetric but heavy-tailed
- Unknown baseline quality

**Example:**
```
Data: [1, 2, 3, 4, 5, 1000]
Median: 3.5, IQR: 2.0 (Q1=2, Q3=4)
Robust score of 1000: 498.25  # Outlier clearly visible!
Robust score of 5: 0.75       # Normal values preserved
```

---

### 3. MAD Scaling (Most Robust)

```python
mad_score = (x - median) / MAD
MAD = median(|x - median(x)|)  # Median Absolute Deviation

# For consistency with std (Gaussian):
scaled_mad = 1.4826 * MAD  # Then std ≈ scaled_MAD for Gaussian
```

**Advantages:**
- 50% breakdown point (most robust possible for location/scale)
- Works for asymmetric distributions
- Consistent with std for Gaussian when scaled by 1.4826

**Risks:**
- Less efficient than std for truly Gaussian data
- MAD can be zero for discrete data with >50% at one value
- Slightly more computational cost

**Use when:**
- Unknown distribution characteristics
- Suspected outliers or anomalies in data
- Industrial/financial data with heavy tails
- Anomaly detection where masking is a concern

**Example:**
```
Data: [1, 2, 3, 4, 5, 1000]
Median: 3.5
Deviations: |1-3.5|=2.5, |2-3.5|=1.5, |3-3.5|=0.5, |4-3.5|=0.5, |5-3.5|=1.5, |1000-3.5|=996.5
MAD: median([2.5, 1.5, 0.5, 0.5, 1.5, 996.5]) = 1.5
Scaled MAD: 1.4826 * 1.5 = 2.22
MAD score of 1000: (1000 - 3.5) / 2.22 = 448.9  # Outlier detected!
```

---

### 4. Min-Max Scaling

```python
scaled = (x - min) / (max - min)  # Scales to [0, 1]
```

**Advantages:**
- Bounded output range
- Preserves distribution shape
- Simple interpretation

**Risks:**
- **Extremely sensitive to outliers** (single outlier affects entire scale)
- No centering - doesn't account for central tendency
- Information loss at boundaries

**Use when:**
- Bounded output is required (neural network input, [0,1] constraints)
- Data is already clean with no outliers
- Distribution shape must be preserved

---

### 5. No Normalization

**Advantages:**
- Preserves actual variance dynamics
- No distributional assumptions
- Useful for detecting physical changes in variance

**Risks:**
- Features with large variance dominate eigenvalues
- Cross-time comparisons difficult
- Scale-dependent results

**Use when:**
- Detecting physical changes in variance is the goal
- Comparing raw covariance structure across conditions
- Features are already on comparable scales

---

## Usage in PRISM

### State Geometry (Eigenvalue Computation)

```python
from prism.entry_points.state_geometry import compute_state_geometry

# Default: z-score (backward compatible)
compute_state_geometry(signal_vector_path, state_vector_path)

# Recommended for industrial data: MAD
compute_state_geometry(signal_vector_path, state_vector_path, norm_method="mad")

# CLI
python -m prism.entry_points.state_geometry -s signal_vector.parquet -t state_vector.parquet --norm mad
```

### Normalization Engine (Direct Use)

```python
from prism.engines.normalization import normalize, recommend_method

# Normalize data
data = np.array([[1, 2], [3, 100], [5, 6]])  # Note outlier

norm_z, params_z = normalize(data, method='zscore')
norm_mad, params_mad = normalize(data, method='mad')
norm_robust, params_robust = normalize(data, method='robust')

# Get method recommendation based on data characteristics
recommendation = recommend_method(data)
print(recommendation)
# {'recommended_method': 'mad', 'reason': 'High outlier fraction (8.3%)', ...}
```

### Anomaly Detection (SQL)

**Z-Score (sensitive to outliers):**
```sql
-- prism/engines/sql/zscore.sql
-- Flags |z| > 3 as anomaly
-- Risk: Outliers inflate std, causing masking
```

**MAD-Based (robust):**
```sql
-- prism/engines/sql/mad_anomaly.sql
-- Flags |mad_score| > 3.5 as anomaly
-- Robust to up to 50% outliers
-- Severity levels: CRITICAL (>5), SEVERE (>3.5), MODERATE (>2.5), MILD (>2)
```

---

## Recommendations by Domain

| Domain | Recommended Method | Reason |
|--------|-------------------|--------|
| **Industrial (pumps, bearings)** | MAD | Heavy tails, unexpected spikes common |
| **Financial (prices, returns)** | MAD or Robust | Fat tails, regime changes |
| **Medical (vitals)** | Robust | Occasional artifacts, bounded ranges |
| **Scientific (controlled)** | Z-Score | Clean data, known Gaussian |
| **Anomaly Detection** | MAD | Avoid masking effect |
| **Regime Detection** | None | Preserve variance dynamics |

---

## Threshold Guidelines

### Z-Score Thresholds (Gaussian assumption)

| Threshold | Probability | Use Case |
|-----------|-------------|----------|
| |z| > 2 | 4.6% | Lenient |
| |z| > 3 | 0.3% | Standard |
| |z| > 4 | 0.006% | Strict |

### MAD Score Thresholds (Robust)

| Threshold | Interpretation | Use Case |
|-----------|---------------|----------|
| |m| > 2.0 | Mild anomaly | Early warning |
| |m| > 2.5 | Moderate | Investigation trigger |
| |m| > 3.5 | Severe | Action required |
| |m| > 5.0 | Critical | Immediate attention |

**Note:** MAD thresholds are slightly higher than z-score because MAD doesn't "absorb" outliers into its scale estimate.

---

## Implementation Files

| File | Purpose |
|------|---------|
| `prism/engines/normalization.py` | Core normalization engine with all methods |
| `prism/entry_points/state_geometry.py` | Uses normalization for eigenvalue computation |
| `prism/engines/sql/zscore.sql` | Z-score anomaly detection (legacy) |
| `prism/engines/sql/mad_anomaly.sql` | MAD-based anomaly detection (robust) |
| `prism/primitives/tests/normalization.py` | Original primitives (z-score, robust, minmax) |

---

## Migration Guide

### From Z-Score to MAD

1. **State Geometry:**
   ```python
   # Before
   compute_state_geometry(sig, state)

   # After
   compute_state_geometry(sig, state, norm_method="mad")
   ```

2. **SQL Anomaly Detection:**
   ```sql
   -- Before: Using zscore.sql with |z| > 3
   -- After: Using mad_anomaly.sql with |mad_score| > 3.5
   ```

3. **Threshold Adjustment:**
   - Z-score threshold of 3 → MAD threshold of ~3.5
   - Z-score threshold of 2 → MAD threshold of ~2.5

---

## References

- Hampel, F. R. (1974). The influence curve and its role in robust estimation. *JASA*
- Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median absolute deviation. *JASA*
- Iglewicz, B., & Hoaglin, D. C. (1993). How to Detect and Handle Outliers. *ASQC Quality Press*
