import numpy                  as     np
import pandas                 as     pd

from   scipy.spatial          import KDTree
from   collections            import defaultdict
from   scipy.stats            import wasserstein_distance
from   scipy.spatial.distance import jensenshannon
from   scipy.spatial.distance import cosine               as sk_cosine_distance

from   src.feature_io         import load_features

def centroid_dist(i, j, centroids):
        return float(np.linalg.norm(centroids[i] - centroids[j]))

def boundary_dist(i, j, boundaries):
    bi, bj = boundaries[i], boundaries[j]

    if len(bi) == 0 or len(bj) == 0:
        return np.nan

    tree_j   = KDTree(bj)
    dists, _ = tree_j.query(bi, k=1)

    return float(np.min(dists))

def bootstrap_ci(data, n_boot=1000, ci=95, seed=42):
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
    valid = real_df.notna().mean() >= min_ratio
    kept  = valid[valid].index.tolist()

    print(f"Keeping {len(kept)} features (≥ {min_ratio*100:.0f}% valid values)")

    return real_df[kept].copy(), synth_df[kept].copy(), kept

def normalize_coverage_features(features):
    mask     = np.isfinite(features).all(axis=1)
    features = features[mask]
    
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    numerator   = features - np.min(features, axis=0)
    denominator = np.max(features, axis=0) - np.min(features, axis=0) + 1e-8

    return numerator / denominator

def normalize_congruence_features(features):
    features    = features[np.isfinite(features)]
    numerator   = features - np.min(features, axis=0)
    denominator = np.max(features, axis=0) - np.min(features, axis=0) + 1e-8

    return numerator / denominator

def combine_features(features_list):
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
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(array), size, replace=replace)

    return array[idx]

def get_cosine_similarity(r, s):
    return float(1 - sk_cosine_distance(r, s))

def get_jensen_shannon_divergence(r, s):
    rn = r / r.sum() if r.sum() else r
    sn = s / s.sum() if s.sum() else s

    return float(jensenshannon(rn, sn, base=2.0)**2)

def get_earth_movers_distance(r, s):
    return float(wasserstein_distance(r, s))

def load_feature_dict(npz_path):
    return load_features(npz_path)

def filter_low_nan_features(real_df, synth_df, min_valid_ratio=0.5):
    valid_mask = real_df.notna().mean() >= min_valid_ratio
    kept       = valid_mask[valid_mask].index.tolist()
    
    print(f"Keeping {len(kept)} / {real_df.shape[1]} features "
          f"(≥ {min_valid_ratio*100:.0f}% valid values)")
          
    return real_df[kept].copy(), synth_df[kept].copy(), kept