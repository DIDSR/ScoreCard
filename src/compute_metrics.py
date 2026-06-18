import numpy                  as     np

from   scipy.stats            import entropy
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

def compute_congruence(r, s, sampling=True, seed=42):
    r = normalize_congruence_features(r)
    s = normalize_congruence_features(s)

    if sampling:
        size = min(len(r), len(s))
        r    = random_sampling(r, size, replace=False, seed=seed)
        s    = random_sampling(s, size, replace=False, seed=seed)

    return {
        'cosine_similarity':         get_cosine_similarity(r, s),
        'jensen_shannon_divergence': get_jensen_shannon_divergence(r, s),
        'earth_movers_distance':     get_earth_movers_distance(r, s)
    }