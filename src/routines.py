import numpy                  as     np
import pandas                 as     pd

from   scipy.spatial          import KDTree
from   collections            import defaultdict
from   scipy.stats            import wasserstein_distance
from   scipy.spatial.distance import jensenshannon
from   scipy.spatial.distance import cosine               as sk_cosine_distance

from   src.feature_io         import load_features

def centroid_dist(i, j, centroids):
    """Compute the Euclidean distance between two centroids.

    Args:
        i: Index of the first centroid.
        j: Index of the second centroid.
        centroids: Array-like of centroid coordinates indexed by i and j.

    Returns:
        float: Euclidean distance between centroids[i] and centroids[j].
    """
    return float(np.linalg.norm(centroids[i] - centroids[j]))

def boundary_dist(i, j, boundaries):
    """Compute the minimum point-to-set distance between two boundary point clouds.

    Args:
        i: Index of the first boundary point set.
        j: Index of the second boundary point set.
        boundaries: Sequence of boundary point arrays indexed by i and j.

    Returns:
        float: Minimum distance from any point in boundaries[i] to boundaries[j],
            or np.nan if either boundary set is empty.
    """
    bi, bj = boundaries[i], boundaries[j]

    if len(bi) == 0 or len(bj) == 0:
        return np.nan

    tree_j   = KDTree(bj)
    dists, _ = tree_j.query(bi, k=1)

    return float(np.min(dists))

def bootstrap_ci(data, n_boot=1000, ci=95, seed=42):
    """Estimate a bootstrap confidence interval for the mean of a 1D sample.

    Args:
        data: Input values; NaNs are removed before resampling.
        n_boot: Number of bootstrap resamples. Defaults to 1000.
        ci: Confidence level in percent. Defaults to 95.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[float, float]: Lower and upper bounds of the confidence interval.
    """
    rng        = np.random.default_rng(seed)
    data       = np.array(data)
    data       = data[~np.isnan(data)]

    boot_means = [
                  np.mean(rng.choice(data, size=len(data), replace=True))
                  for _ in range(n_boot)
                 ]

    lower      = np.percentile(boot_means, (100 - ci) / 2)
    upper      = np.percentile(boot_means, 100 - (100 - ci) / 2)

    return lower, upper

def flatten_features(features_list):
    """Merge a list of per-sample feature dicts into a single dict of lists.

    Args:
        features_list: List of dicts mapping feature names to scalar or
            sequence values.

    Returns:
        dict: Feature names mapped to concatenated value lists across samples.
    """
    all_features = defaultdict(list)
    
    for feat_dict in features_list:
        for key in feat_dict:
            val = feat_dict[key]
            
            if np.isscalar(val):
                all_features[key].append(val)
            else:
                all_features[key].extend(val)
    
    return all_features

def filter_features(real_df, synth_df, min_ratio=0.5):
    """Drop columns with insufficient non-NaN values in the real feature DataFrame.

    Args:
        real_df: DataFrame of real-image features used to determine validity.
        synth_df: DataFrame of synthetic-image features filtered to the same columns.
        min_ratio: Minimum fraction of non-NaN values required per column.
            Defaults to 0.5.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list]: Filtered real_df, filtered
            synth_df, and list of retained column names.
    """
    valid = real_df.notna().mean() >= min_ratio
    kept  = valid[valid].index.tolist()

    print(f"Keeping {len(kept)} features (≥ {min_ratio*100:.0f}% valid values)")

    return real_df[kept].copy(), synth_df[kept].copy(), kept

def normalize_coverage_features(features):
    """Min-max normalize coverage features along each column.

    Rows containing non-finite values are removed before normalization.

    Args:
        features: 1D or 2D array of coverage feature values.

    Returns:
        np.ndarray: Column-wise min-max scaled features in [0, 1].
    """
    mask     = np.isfinite(features).all(axis=1)
    features = features[mask]
    
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    numerator   = features - np.min(features, axis=0)
    denominator = np.max(features, axis=0) - np.min(features, axis=0) + 1e-8

    return numerator / denominator

def normalize_congruence_features(features):
    """Min-max normalize congruence features along each column.

    Non-finite values are removed before normalization.

    Args:
        features: Array of congruence feature values.

    Returns:
        np.ndarray: Column-wise min-max scaled features in [0, 1].
    """
    features    = features[np.isfinite(features)]
    numerator   = features - np.min(features, axis=0)
    denominator = np.max(features, axis=0) - np.min(features, axis=0) + 1e-8

    return numerator / denominator

def combine_features(features_list):
    """Horizontally concatenate feature arrays loaded from multiple NPZ files.

    Each file is truncated to the minimum shared sample length before merging.

    Args:
        features_list: Ordered list of paths to .npz feature files.

    Returns:
        pd.DataFrame: Combined feature matrix with columns from all inputs.
    """
    combined_dict   = load_features(features_list[0])
    current_min_len = min(len(v) for v in combined_dict.values())
    combined_dict   = {k: v[:current_min_len] for k, v in combined_dict.items()}
    combined_pd     = pd.DataFrame(combined_dict)

    for sub_feature_path in features_list[1:]:
        sub_dict        = load_features(sub_feature_path)
        min_len         = min(min(len(v) for v in sub_dict.values()),
                              min(len(v) for v in combined_dict.values()))

        sub_dict        = {k: v[:min_len] for k, v in sub_dict.items()}
        combined_dict   = {k: v[:min_len] for k, v in combined_dict.items()}

        combined_pd     = pd.concat([pd.DataFrame(combined_dict), pd.DataFrame(sub_dict)], axis=1)
        combined_dict   = combined_pd.to_dict(index=False)
    
    return combined_pd

def random_sampling(array, size, replace=False, seed=None):
    """Randomly sample elements from an array by index.

    Args:
        array: Input array to sample from.
        size: Number of elements to draw.
        replace: Whether sampling is with replacement. Defaults to False.
        seed: Optional random seed for reproducibility.

    Returns:
        np.ndarray: Subset of array at the sampled indices.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(array), size, replace=replace)

    return array[idx]

def get_cosine_similarity(r, s):
    """Compute cosine similarity between two vectors.

    Args:
        r: First vector.
        s: Second vector.

    Returns:
        float: Cosine similarity in [0, 1], where 1 indicates identical direction.
    """
    return float(1 - sk_cosine_distance(r, s))

def get_jensen_shannon_divergence(r, s):
    """Compute the squared Jensen-Shannon divergence between two distributions.

    Inputs are normalized to probability mass functions before comparison.

    Args:
        r: First distribution or count vector.
        s: Second distribution or count vector.

    Returns:
        float: Squared Jensen-Shannon divergence with base-2 logarithm.
    """
    rn = r / r.sum() if r.sum() else r
    sn = s / s.sum() if s.sum() else s

    return float(jensenshannon(rn, sn, base=2.0)**2)

def get_earth_movers_distance(r, s):
    """Compute the Earth Mover's (Wasserstein) distance between two 1D distributions.

    Args:
        r: First 1D distribution or sample.
        s: Second 1D distribution or sample.

    Returns:
        float: Wasserstein distance between r and s.
    """
    return float(wasserstein_distance(r, s))

def load_feature_dict(npz_path):
    """Load a feature dictionary from an NPZ file.

    Args:
        npz_path: Path to the .npz feature archive.

    Returns:
        dict: Feature names mapped to NumPy arrays.
    """
    return load_features(npz_path)

def filter_low_nan_features(real_df, synth_df, min_valid_ratio=0.5):
    """Drop feature columns with too many NaN values in the real DataFrame.

    Args:
        real_df: DataFrame of real-image features used to determine validity.
        synth_df: DataFrame of synthetic-image features filtered to the same columns.
        min_valid_ratio: Minimum fraction of non-NaN values required per column.
            Defaults to 0.5.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list]: Filtered real_df, filtered
            synth_df, and list of retained column names.
    """
    valid_mask = real_df.notna().mean() >= min_valid_ratio
    kept       = valid_mask[valid_mask].index.tolist()
    
    print(f"Keeping {len(kept)} / {real_df.shape[1]} features "
          f"(≥ {min_valid_ratio*100:.0f}% valid values)")
          
    return real_df[kept].copy(), synth_df[kept].copy(), kept