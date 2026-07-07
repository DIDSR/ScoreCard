import numpy                  as     np
import pandas                 as     pd

from   scipy.stats            import entropy, f_oneway
from   scipy.spatial          import ConvexHull
from   sklearn.decomposition  import PCA
from   sklearn.metrics        import pairwise_distances
from   src.routines           import normalize_coverage_features, normalize_congruence_features, random_sampling, get_cosine_similarity, get_earth_movers_distance, get_jensen_shannon_divergence

def compute_coverage(df, label=""):
    """Compute coverage metrics that quantify spread and diversity of feature values.

    Normalizes input features, then computes variance, entropy, mean distance to
    the centroid, and convex hull volume (when at least two features are present).

    Args:
        df: DataFrame of feature values; each row is a sample and each column is a
            feature.
        label: Optional label used in warning messages and printed output.
            Defaults to an empty string.

    Returns:
        dict: Coverage metrics with keys ``Variance``, ``Entropy``,
        ``Distance_to_Centroid``, and ``Convex_Hull_Volume``. Values are ``np.nan``
        when there are fewer than 10 samples or a metric cannot be computed.
    """
    arr = df.to_numpy()

    if len(arr) < 10:
        print(f"  Warning: Too few samples for {label}")
        return {k: np.nan for k in ['Variance', 'Entropy', 'Distance_to_Centroid', 'Convex_Hull_Volume']}

    arr_norm   = normalize_coverage_features(arr)
    n_features = arr_norm.shape[1]
    vector     = np.mean(arr_norm, axis=0)
    centroid   = np.mean(arr_norm, axis=0).reshape(1, -1)

    if n_features >= 2:
        reduced         = PCA(n_components=2).fit_transform(arr_norm)
        convex_hull_vol = ConvexHull(reduced).volume

    else:
        convex_hull_vol = np.nan

    metrics = {
        'Variance':             np.var(arr_norm),
        'Entropy':              entropy(vector / np.sum(vector)) if n_features > 1 else np.nan,
        'Distance_to_Centroid': np.mean(pairwise_distances(arr_norm, centroid)),
        'Convex_Hull_Volume':   convex_hull_vol
    }

    for k, v in metrics.items():
        val_str = f"{v:.6f}" if not np.isnan(v) else "N/A (single feature)"
        print(f"  {k:22s}: {val_str}")

    return metrics

def compute_congruence(metrics_to_compute, r, s, sampling=True, seed=42):
    """Compute distributional similarity metrics between two feature sets.

    Optionally subsamples both sets to the same size before computing the
    requested metrics.

    Args:
        metrics_to_compute: Dict of flags indicating which metrics to compute.
            Supported keys are ``'jsd'``, ``'emd'``, and ``'cosine'``.
        r: Real feature values, array-like.
        s: Synthetic feature values, array-like.
        sampling: If True, randomly subsample ``r`` and ``s`` to the same size
            before comparison. Defaults to True.
        seed: Random seed used when ``sampling`` is True. Defaults to 42.

    Returns:
        dict: Requested similarity metrics. May include
        ``'jensen_shannon_divergence'``, ``'earth_movers_distance'``, and
        ``'cosine_similarity'`` depending on ``metrics_to_compute``.
    """
    return_metrics = {}
    r              = normalize_congruence_features(r)
    s              = normalize_congruence_features(s)

    if sampling:
        size = min(len(r), len(s))
        r    = random_sampling(r, size, replace=False, seed=seed)
        s    = random_sampling(s, size, replace=False, seed=seed)

    if metrics_to_compute['jsd']    == True:
        return_metrics['jensen_shannon_divergence'] = get_jensen_shannon_divergence(r, s)

    if metrics_to_compute['emd']    == True:
        return_metrics['earth_movers_distance']     = get_earth_movers_distance(r, s)

    if metrics_to_compute['cosine'] == True:
        return_metrics['cosine_similarity']         = get_cosine_similarity(r, s)

    return return_metrics

def compute_completeness(df, required_fields=None, label=""):
    """Compute missing-data and required-field completeness metrics.

    Args:
        df: DataFrame or array-like object containing feature values. For
            DataFrames, missing values are detected with ``isna()``.
        required_fields: Column names to evaluate for required-field completeness.
            If None, all DataFrame columns are used. Ignored for non-DataFrame
            inputs.
        label: Optional label used in warning messages and printed output.
            Defaults to an empty string.

    Returns:
        dict: Completeness metrics with keys ``Missing_Data_Percentage``,
        ``Required_Fields_Completeness``, and ``per_field``. ``per_field`` maps
        each required column to the percentage of non-missing values. Values are
        ``np.nan`` when the input is empty or a metric cannot be computed.
    """
    if df is None or len(df) == 0:
        print(f"  Warning: Too few samples for {label}")
        return {k: np.nan for k in ['Missing_Data_Percentage', 'Required_Fields_Completeness']}

    if hasattr(df, 'isna'):
        missing = int(df.isna().sum().sum())
        total   = int(df.size)

        if required_fields is None:
            req_fields = list(df.columns) if hasattr(df, 'columns') else None
        else:
            req_fields = [f for f in (required_fields if isinstance(required_fields, (list, tuple)) else [required_fields])
                          if f in df.columns]

        if req_fields:
            try:
                req_df           = df[req_fields]
                req_missing      = int(req_df.isna().sum().sum())
                req_total        = int(req_df.size)
                req_complete_pct = ((req_total - req_missing) / req_total * 100.0) if req_total > 0 else np.nan

                per_field = {}

                for col in req_fields:
                    n_total   = len(df)
                    n_present = int(df[col].notna().sum())
                    per_field[col] = (n_present / n_total * 100.0) if n_total > 0 else np.nan

            except Exception:
                req_complete_pct = np.nan
                per_field        = {}
        else:
            req_complete_pct = np.nan
            per_field        = {}
    else:
        arr   = np.asarray(df)
        total = arr.size

        try:
            missing = int(np.sum(np.isnan(arr)))
        except (TypeError, ValueError):
            missing = 0
        req_complete_pct = ((total - missing) / total * 100.0) if total > 0 else np.nan
        per_field        = {}

    missing_pct = (missing / total * 100.0) if total > 0 else np.nan

    metrics = {
        'Missing_Data_Percentage':      missing_pct,
        'Required_Fields_Completeness': req_complete_pct,
        'per_field':                    per_field
    }

    for k, v in metrics.items():
        if k == 'per_field':
            continue
        val_str = f"{v:.2f}%" if not np.isnan(v) else "N/A"
        print(f"  {k:30s}: {val_str}")

    return metrics

def compute_consistency(group_data, label=""):
    """Compute consistency metrics across subgroups of metric values.

    Compares subgroup means using variance across groups, max-min spread, and a
    one-way ANOVA test for significant differences.

    Args:
        group_data: Dict mapping subgroup names to array-like metric values for
            that subgroup, e.g. ``{group_name: values}``. At least two non-empty
            groups are required.
        label: Optional label used in warning messages and printed output.
            Defaults to an empty string.

    Returns:
        dict: Consistency metrics with keys ``Variance_of_Group_Means``,
        ``Max_Min_Difference``, ``ANOVA_F_statistic``, and ``ANOVA_p_value``.
        Values are ``np.nan`` when there are fewer than two groups, a group is
        empty, or ANOVA fails.
    """
    if not isinstance(group_data, dict) or len(group_data) < 2:
        print(f"  Warning: Too few groups (<2) for consistency computation in {label}")
        return {k: np.nan for k in ['Variance_of_Group_Means', 'Max_Min_Difference', 'ANOVA_F_statistic', 'ANOVA_p_value']}

    groups     = list(group_data.keys())
    data_lists = []

    for g in groups:
        d = np.asarray(group_data[g], dtype=float).flatten()

        if len(d) == 0:
            print(f"  Warning: Empty data for group '{g}' in {label}")
            return {k: np.nan for k in ['Variance_of_Group_Means', 'Max_Min_Difference', 'ANOVA_F_statistic', 'ANOVA_p_value']}
        data_lists.append(d)

    group_means  = np.array([np.mean(d) for d in data_lists])
    n_groups     = len(group_means)

    var_across   = np.var(group_means, ddof=1) if n_groups > 1 else np.nan
    max_min_diff = np.max(group_means) - np.min(group_means) if n_groups > 1 else np.nan

    try:
        f_stat, p_val = f_oneway(*data_lists)
    except Exception as e:
        f_stat, p_val = np.nan, np.nan
        print(f"  Warning: ANOVA failed for {label}: {e}")

    metrics = {
        'Variance_of_Group_Means':   var_across,
        'Max_Min_Difference':        max_min_diff,
        'ANOVA_F_statistic':         f_stat,
        'ANOVA_p_value':             p_val
    }

    for k, v in metrics.items():
        if k == 'ANOVA_p_value':
            val_str = f"{v:.4g}" if not np.isnan(v) else "N/A"
        elif k == 'ANOVA_F_statistic':
            val_str = f"{v:.4f}" if not np.isnan(v) else "N/A"
        else:
            val_str = f"{v:.6f}" if not np.isnan(v) else "N/A"
        print(f"  {k:25s}: {val_str}")

    return metrics

def compute_constraint(real_features, synth_features, features_to_check=None, percentile_low=1, percentile_high=99):
    """Measure how often synthetic feature values violate real-data percentile bounds.

    For each feature, defines lower and upper bounds from the real distribution
    and reports the percentage of synthetic values that fall outside that range.

    Args:
        real_features: Dict mapping feature names to real feature value arrays.
        synth_features: Dict mapping feature names to synthetic feature value
            arrays.
        features_to_check: Feature names to evaluate. If None, automatically
            selects common numeric 1-D features present in both dicts with more
            than 10 real values.
        percentile_low: Lower percentile used to define the real-data bound.
            Defaults to 1.
        percentile_high: Upper percentile used to define the real-data bound.
            Defaults to 99.

    Returns:
        pd.DataFrame: One row per evaluated feature with columns ``Feature``,
        ``Synth_Violation_%``, ``Lower_Bound``, ``Upper_Bound``, ``n_real``, and
        ``n_synth``. Rows are sorted by ``Synth_Violation_%`` in descending order.
        Returns an empty DataFrame when no features qualify.
    """
    if features_to_check is None:
        common_keys       = set(real_features.keys()) & set(synth_features.keys())
        features_to_check = []

        for k in sorted(common_keys):
            try:
                arr = np.asarray(real_features[k], dtype=float)
                
                if arr.ndim == 1 and len(arr) > 10 and np.issubdtype(arr.dtype, np.number):
                    features_to_check.append(k)
            except:
                continue
    
    results = []
    
    for feat in features_to_check:
        if feat not in real_features or feat not in synth_features:
            continue
            
        real_vals   = np.asarray(real_features[feat], dtype=float)
        synth_vals  = np.asarray(synth_features[feat], dtype=float)
        
        real_clean  = real_vals[~np.isnan(real_vals)]
        synth_clean = synth_vals[~np.isnan(synth_vals)]
        
        if len(real_clean) < 10 or len(synth_clean) < 10:
            continue
            
        lower      = np.nanpercentile(real_clean, percentile_low)
        upper      = np.nanpercentile(real_clean, percentile_high)
        synth_viol = np.mean((synth_clean < lower) | (synth_clean > upper)) * 100
        
        results.append({
            'Feature'          : feat,
            'Synth_Violation_%': round(synth_viol, 2),
            'Lower_Bound'      : round(lower, 4),
            'Upper_Bound'      : round(upper, 4),
            'n_real'           : len(real_clean),
            'n_synth'          : len(synth_clean)
        })
    
    violation_df = pd.DataFrame(results)

    if not violation_df.empty:
        violation_df = violation_df.sort_values('Synth_Violation_%', ascending=False).reset_index(drop=True)
    
    return violation_df
