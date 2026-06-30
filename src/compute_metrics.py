import numpy                  as     np
import pandas                 as     pd

from   scipy.stats            import entropy, f_oneway
from   scipy.spatial          import ConvexHull
from   sklearn.decomposition  import PCA
from   sklearn.metrics        import pairwise_distances
from   src.routines           import normalize_coverage_features, normalize_congruence_features, random_sampling, get_cosine_similarity, get_earth_movers_distance, get_jensen_shannon_divergence

def compute_coverage(df, label=""):
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
    """Compute consistency metrics across subgroups: variance of group means,
    max-min difference of group means, and one-way ANOVA for significance of differences.
    group_data should be a dict: {group_name: array-like of metric values for that subgroup}
    Follows style of other compute_* functions.
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
