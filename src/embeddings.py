"""Dimensionality-reduction helpers for exploratory congruence visualizations.

PCA / t-SNE / UMAP projections are diagnostic overlays of joint feature space.
They are not ScoreCard scalar metrics (unlike JSD, EMD, or cosine similarity).
"""

from __future__ import annotations

import numpy  as np
import pandas as pd

from   sklearn.decomposition import PCA
from   sklearn.manifold      import TSNE
from   sklearn.preprocessing import MinMaxScaler

from src.routines import random_sampling

try:
    import umap as umap_lib
except ImportError:
    umap_lib = None


REAL_COLOR  = "#9999CC"
SYNTH_COLOR = "#FF9966"


def prepare_feature_matrices(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_names: list[str] | None = None,
    *,
    balance: bool = False,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build aligned finite real/synthetic matrices for the same feature columns.

    Args:
        real_df: Real-sample feature table.
        synth_df: Synthetic-sample feature table.
        feature_names: Optional column subset. Defaults to columns shared by both.
        balance: If True, subsample both sets to the same size (min length).
        max_samples: Optional per-class cap after cleaning. Useful for large
            nucleus feature sets before t-SNE / UMAP. Applied independently to
            real and synthetic unless ``balance`` is True (then both use
            ``min(len_real, len_synth, max_samples)``).
        seed: Random seed used when subsampling.

    Returns:
        Tuple of ``(X_real, X_synth, cols)`` as float arrays and the column list.
        Rows with any non-finite value are dropped.
    """
    if feature_names is None:
        cols = [c for c in real_df.columns if c in synth_df.columns]
    else:
        cols = [c for c in feature_names if c in real_df.columns and c in synth_df.columns]

    if not cols:
        return np.empty((0, 0)), np.empty((0, 0)), []

    X_real     = real_df[cols].to_numpy(dtype=float)
    X_synth    = synth_df[cols].to_numpy(dtype=float)

    real_mask  = np.isfinite(X_real).all(axis=1)
    synth_mask = np.isfinite(X_synth).all(axis=1)
    X_real     = X_real[real_mask]
    X_synth    = X_synth[synth_mask]

    if balance and len(X_real) > 0 and len(X_synth) > 0:
        size = min(len(X_real), len(X_synth))

        if max_samples is not None:
            size = min(size, int(max_samples))

        X_real  = random_sampling(X_real, size, replace=False, seed=seed)
        X_synth = random_sampling(X_synth, size, replace=False, seed=seed)

    elif max_samples is not None:
        cap = int(max_samples)

        if len(X_real) > cap:
            X_real = random_sampling(X_real, cap, replace=False, seed=seed)

        if len(X_synth) > cap:
            X_synth = random_sampling(X_synth, cap, replace=False, seed=seed + 1)

    return X_real, X_synth, cols


def _scale_for_pca(X_real: np.ndarray, X_synth: np.ndarray):
    """Min-max scale using the real distribution only, then transform both."""
    scaler    = MinMaxScaler()
    X_real_s  = scaler.fit_transform(X_real)
    X_synth_s = scaler.transform(X_synth)

    return X_real_s, X_synth_s, scaler


def _scale_combined(X_real: np.ndarray, X_synth: np.ndarray):
    """Min-max scale fit on the stacked real+synthetic matrix."""
    scaler  = MinMaxScaler()
    X_all   = np.vstack([X_real, X_synth])
    X_all_s = scaler.fit_transform(X_all)
    n_real  = len(X_real)

    return X_all_s[:n_real], X_all_s[n_real:], scaler


def compute_pca_embedding(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    *,
    n_components: int = 2,
    random_state: int = 42,
) -> dict:
    """Project real and synthetic features with PCA fit on real samples only.

    Fitting on real data and transforming synthetic samples makes domain shift
    more visible when synthetic points leave the real span.
    """
    del random_state
    n_components = min(n_components, X_real.shape[0], X_real.shape[1])

    if n_components < 1:
        raise ValueError("Need at least one finite real sample and one feature for PCA.")

    X_real_s, X_synth_s, _ = _scale_for_pca(X_real, X_synth)

    pca       = PCA(n_components=n_components)
    real_emb  = pca.fit_transform(X_real_s)
    synth_emb = pca.transform(X_synth_s)

    method = "pca_1d" if n_components == 1 else "pca"

    title = (
        "PCA 1D — PC1 (fit on real, project both)"
        if n_components == 1
        else "PCA (fit on real, project both)"
    )

    return {
        "method": method,
        "real"  : real_emb,
        "synth" : synth_emb,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "axis_labels": [
            f"PC{i + 1} ({100 * r:.1f}% var)"
            for i, r in enumerate(pca.explained_variance_ratio_)
        ],
        "title": title,
        "note" : "",
    }


def compute_pca_1d_embedding(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    *,
    random_state: int = 42,
) -> dict:
    """1D PCA projection onto PC1 (fit on real, project both)."""
    return compute_pca_embedding(
        X_real, X_synth, n_components=1, random_state=random_state
    )


def compute_tsne_embedding(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    *,
    n_components: int = 2,
    perplexity: float | None = None,
    random_state: int = 42,
    pca_preprocess_dims: int = 50,
) -> dict:
    """Joint t-SNE embedding of stacked real and synthetic features."""
    X_real_s, X_synth_s, _ = _scale_combined(X_real, X_synth)

    X_all = np.vstack([X_real_s, X_synth_s])
    n     = len(X_all)

    if n < 3:
        raise ValueError(f"t-SNE needs at least 3 samples; got {n}.")

    max_pca = min(pca_preprocess_dims, X_all.shape[1], n - 1)

    if max_pca >= 2 and X_all.shape[1] > max_pca:
        X_all = PCA(n_components=max_pca, random_state=random_state).fit_transform(X_all)

    if perplexity is None:
        perplexity = min(30.0, max(5.0, (n - 1) / 3.0))

    perplexity = float(min(perplexity, n - 1))

    if perplexity < 1.0:
        raise ValueError(f"Invalid t-SNE perplexity {perplexity} for n={n}.")

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    emb    = tsne.fit_transform(X_all)
    n_real = len(X_real_s)

    return {
        "method": "tsne",
        "real": emb[:n_real],
        "synth": emb[n_real:],
        "explained_variance_ratio": None,
        "axis_labels": [f"t-SNE {i + 1}" for i in range(n_components)],
        "title": f"t-SNE (perplexity={perplexity:.1f})",
        "note": "",
        "perplexity": perplexity,
    }


def compute_umap_embedding(
    X_real : np.ndarray,
    X_synth: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int | None = None,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Joint UMAP embedding of stacked real and synthetic features."""
    if umap_lib is None:
        raise ImportError(
            "umap-learn is not installed."
        )

    X_real_s, X_synth_s, _ = _scale_combined(X_real, X_synth)
    X_all                  = np.vstack([X_real_s, X_synth_s])
    n                      = len(X_all)

    if n < 3:
        raise ValueError(f"UMAP needs at least 3 samples; got {n}.")

    if n_neighbors is None:
        n_neighbors = min(15, n - 1)

    n_neighbors = int(max(2, min(n_neighbors, n - 1)))

    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    emb    = reducer.fit_transform(X_all)
    n_real = len(X_real_s)

    return {
        "method": "umap",
        "real": emb[:n_real],
        "synth": emb[n_real:],
        "explained_variance_ratio": None,
        "axis_labels": [f"UMAP {i + 1}" for i in range(n_components)],
        "title": f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})",
        "note": "",
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
    }


_EMBEDDING_FNS = {
    "pca"   : compute_pca_embedding,
    "pca_1d": compute_pca_1d_embedding,
    "tsne"  : compute_tsne_embedding,
    "umap"  : compute_umap_embedding,
}


def compute_embeddings(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    methods: dict[str, bool] | None = None,
    *,
    random_state: int = 42,
) -> dict[str, dict]:
    """Compute selected embeddings for real vs synthetic feature matrices.

    Args:
        X_real: Real feature matrix ``(n_real, n_features)``.
        X_synth: Synthetic feature matrix ``(n_synth, n_features)``.
        methods: Mapping of method name to enable flag. Supported keys:
            ``pca``, ``pca_1d``, ``tsne``, ``umap``. Defaults to all True.
        random_state: Seed for stochastic methods.

    Returns:
        Dict mapping method name to embedding result dicts. Methods that fail
        (missing optional deps, too few samples) are omitted with a printed warning.
    """
    if methods is None:
        methods = {"pca": True, "pca_1d": True, "tsne": True, "umap": True}

    if X_real.size == 0 or X_synth.size == 0:
        print("  Warning: empty feature matrices; skipping embeddings.")

        return {}

    results: dict[str, dict] = {}

    for name, enabled in methods.items():
        if not enabled:
            continue

        fn = _EMBEDDING_FNS.get(name)

        if fn is None:
            print(f"  Warning: unknown embedding method '{name}', skipping.")
            continue
        
        try:
            results[name] = fn(X_real, X_synth, random_state=random_state)
            print(f"  {name}: real={len(results[name]['real'])}, synth={len(results[name]['synth'])}")
        except ImportError as e:
            print(f"  Warning: {e}")
        except Exception as e:
            print(f"  Warning: {name} embedding failed: {e}")

    return results
