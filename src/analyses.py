import os

import pandas                as     pd
import numpy                 as     np

from   src.feature_io        import load_features
from   src.routines          import combine_features, filter_features
from   src.compute_metrics   import (
                                        compute_congruence, 
                                        compute_coverage,
                                        compute_completeness,
                                        compute_consistency,
                                        compute_constraint,
                                    )
from   src.embeddings        import prepare_feature_matrices, compute_embeddings

def hist_analysis(results_dir="./data/features", feature_names=None):
    """Build overlaid histogram figures comparing real and synthetic patch features.

    Args:
        results_dir: Directory containing real and synthetic patch feature NPZ files.
        feature_names: Optional list of feature keys to include. When omitted, all
            features present in both datasets are used.

    Returns:
        A tuple ``(fig_hist, None)`` where ``fig_hist`` is the histogram figure
        produced by ``print_histograms``.
    """
    from src.visualization import print_histograms

    real_features  = load_features(os.path.join(results_dir, 'real_patch_appearance_features.npz'))
    synth_features = load_features(os.path.join(results_dir, 'static_patch_appearance_features.npz'))

    if feature_names:
        real_features  = {k: real_features[k] for k in feature_names if k in real_features}
        synth_features = {k: synth_features[k] for k in feature_names if k in synth_features}

    fig_hist = print_histograms(real_features, synth_features)

    return fig_hist, None


def coverage_analysis(
    results_dir='./data/features',
    real_features='real_patch_appearance_features.npz',
    synth_features='kde_patch_appearance_features.npz',
    feature_names=None,
    metrics_to_compute=None,
):
    """Compute and plot coverage metrics for real vs. synthetic patch features.

    Args:
        results_dir: Directory containing patch feature NPZ files.
        real_features: Filename of the real-data feature NPZ within ``results_dir``.
        synth_features: Filename of the synthetic-data feature NPZ within
            ``results_dir``.
        feature_names: Optional subset of feature column names to analyze after
            variance-based filtering.
        metrics_to_compute: Optional mapping of metric name to bool; only metrics
            marked ``True`` are plotted.

    Returns:
        A dict mapping each coverage metric name to its bar-plot figure. Returns
        an empty dict when no features remain after filtering.
    """
    real_features                    = os.path.join(results_dir, real_features)
    synth_features                   = os.path.join(results_dir, synth_features)

    real_features_list               = []
    synth_features_list              = []

    real_features_list.append(real_features)
    synth_features_list.append(synth_features)

    real_df                          = combine_features(real_features_list)
    synth_df                         = combine_features(synth_features_list)

    real_df, synth_df, kept_features = filter_features(real_df, synth_df, 0.5)
    cols = kept_features

    if feature_names:
        cols = [c for c in cols if c in feature_names]

    if not cols:
        return {}

    real_features_coverage  = compute_coverage(real_df[cols], "Real")
    synth_features_coverage = compute_coverage(synth_df[cols], "Synth")

    coverage_df = pd.DataFrame({
        'Real_Features':  real_features_coverage,
        'Synth_Features': synth_features_coverage,
    }).T

    from src.visualization import create_barplot

    df_long = (
        coverage_df.reset_index()
        .rename(columns={"index": "Dataset"})
        .melt(id_vars="Dataset", var_name="Metric", value_name="Value")
    )

    if metrics_to_compute:
        enabled = [m for m, on in metrics_to_compute.items() if on]
        df_long = df_long[df_long["Metric"].isin(enabled)]

    figs = {}
    for metric in df_long["Metric"].unique():
        sub = df_long[df_long["Metric"] == metric]
        pretty = metric.replace("_", " ").title()
        figs[metric] = create_barplot(
            sub,
            x="Value",
            y="Dataset",
            hue="Dataset",
            suptitle=f"Coverage — {pretty}",
            xlabel="Value",
            bar_label_fmt="%.4f",
            sort=False,
        )

    return figs

def congruence_analysis(
    metrics_to_compute,
    results_dir='./data/features',
    real_features='real_patch_appearance_features.npz',
    synth_features='kde_patch_appearance_features.npz',
    feature_names=None,
):
    """Compute congruence metrics between real and synthetic patch feature distributions.

    Args:
        metrics_to_compute: Mapping of metric identifiers (e.g. ``jsd``, ``emd``,
            ``cosine``) to booleans indicating which metrics to compute.
        results_dir: Directory containing patch feature NPZ files.
        real_features: Filename of the real-data feature NPZ within ``results_dir``.
        synth_features: Filename of the synthetic-data feature NPZ within
            ``results_dir``.
        feature_names: Optional subset of feature column names to analyze after
            variance-based filtering.

    Returns:
        A dict mapping each congruence metric name to a bar-plot figure comparing
        features.
    """
    real_features        = os.path.join(results_dir, real_features)
    synth_features       = os.path.join(results_dir, synth_features)

    real_features_list   = []
    synth_features_list  = []

    real_features_list.append(real_features)
    synth_features_list.append(synth_features)

    real_df              = combine_features(real_features_list)
    synth_df             = combine_features(synth_features_list)

    real_df, synth_df, kept_features = filter_features(real_df, synth_df, 0.5)

    congruence_results = {}
    cols = kept_features
    if feature_names:
        cols = [c for c in cols if c in feature_names]

    for feature in cols:
        r                           = real_df[feature].values
        s                           = synth_df[feature].values
        congruence_results[feature] = compute_congruence(metrics_to_compute, r, s, sampling=True, seed=42)

    metric_column_map = {
        'cosine_similarity'        : 'Cosine_Similarity',
        'jensen_shannon_divergence': 'JSD',
        'earth_movers_distance'    : 'EMD_Wasserstein',
    }

    summary_list = []

    for feature, values in congruence_results.items():
        row = {
            'Synthetic': 'Synth',
            'Real'     : 'Real',
            'Feature'  : feature,
        }
        for result_key, column_name in metric_column_map.items():
            if result_key in values:
                row[column_name] = values[result_key]
        summary_list.append(row)

    congruence_df  = pd.DataFrame(summary_list)

    from src.visualization import create_barplot
    metric_cols = [c for c in congruence_df.columns if c not in ("Feature", "Synthetic", "Real")]
    df_melt = congruence_df.melt(
        id_vars=["Feature"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Value"
    )

    figs = {}
    for metric in df_melt["Metric"].unique():
        sub = df_melt[df_melt["Metric"] == metric]
        pretty = metric.replace("_", " ").title()
        figs[metric] = create_barplot(
            sub,
            x="Value",
            y="Feature",
            suptitle=f"Real vs. Synthetic {pretty}",
            sort=True,
            ascending=True,
            bar_label_fmt="%.4f",
        )

    return figs


def embedding_analysis(
    methods=None,
    results_dir='./data/features',
    real_features='real_nucleus_appearance_features.npz',
    synth_features='kde_synth_nucleus_appearance_features.npz',
    feature_names=None,
    *,
    balance=False,
    max_samples=5000,
    seed=42,
):
    """Produce exploratory PCA / t-SNE / UMAP plots for real vs synthetic features.

    Defaults to nucleus appearance NPZs. Intended for the embeddings notebook
    (``09_Embeddings.ipynb``):

    Args:
        methods: Mapping of embedding method name to bool. Supported keys are
            ``pca``, ``pca_1d``, ``tsne``, and ``umap``. Defaults to all enabled.
            ``pca_1d`` is a density plot of the first principal component.
        results_dir: Directory containing feature NPZ files.
        real_features: Filename of the real-data feature NPZ within ``results_dir``.
            Defaults to ``real_nucleus_appearance_features.npz``.
        synth_features: Filename of the synthetic-data feature NPZ within
            ``results_dir``. Defaults to
            ``kde_synth_nucleus_appearance_features.npz``.
        feature_names: Optional subset of feature column names after filtering.
        balance: If True, subsample real and synthetic to the same count before
            embedding.
        max_samples: Per-class sample cap before embedding (default 5000). Keeps
            t-SNE / UMAP tractable on large nucleus sets (~10k–90k). Pass
            ``None`` to use all finite samples.
        seed: Random seed for balancing and stochastic embeddings.

    Returns:
        A dict mapping embedding method names to matplotlib figures. Empty when
        no features remain or all methods fail.
    """
    if methods is None:
        methods = {"pca": True, "pca_1d": True, "tsne": True, "umap": True}

    real_path  = os.path.join(results_dir, real_features)
    synth_path = os.path.join(results_dir, synth_features)

    real_df    = combine_features([real_path])
    synth_df   = combine_features([synth_path])

    real_df, synth_df, kept_features = filter_features(real_df, synth_df, 0.5)
    cols                             = kept_features

    if feature_names:
        cols = [c for c in cols if c in feature_names]

    if not cols:
        print("  Warning: no features left for embedding analysis.")
        return {}

    X_real, X_synth, cols = prepare_feature_matrices(
        real_df[cols],
        synth_df[cols],
        feature_names=cols,
        balance=balance,
        max_samples=max_samples,
        seed=seed,
    )

    if X_real.size == 0 or X_synth.size == 0:
        print("  Warning: empty matrices after cleaning; skipping embeddings.")
        return {}

    print(f"Embedding analysis on {len(cols)} features "
          f"(real n={len(X_real)}, synth n={len(X_synth)}; "
          f"max_samples={max_samples})")

    embeddings = compute_embeddings(X_real, X_synth, methods=methods, random_state=seed)

    from src.visualization import plot_embedding_1d, plot_embedding_scatter

    figs = {}

    for name, emb in embeddings.items():
        real  = emb.get("real")
        is_1d = (
            emb.get("method") == "pca_1d"
            or name == "pca_1d"
            or (real is not None and getattr(real, "ndim", 0) == 2 and real.shape[1] == 1)
        )
        
        figs[name] = plot_embedding_1d(emb) if is_1d else plot_embedding_scatter(emb)

    return figs


def completeness_analysis(real_csv, synth_csv, required_fields=None, label="", metrics_to_include=None):
    """Measure metadata completeness for real and synthetic CSV datasets.

    Args:
        real_csv: Path to the real-data metadata CSV.
        synth_csv: Path to the synthetic-data metadata CSV.
        required_fields: Column names to treat as required. Defaults to columns
            shared by both CSVs.
        label: Prefix applied to internal completeness computation labels.
        metrics_to_include: Optional mapping of metric name to bool controlling
            which summary and per-field plots are generated.

    Returns:
        A tuple ``(comp_df, figs)`` where ``comp_df`` is a summary DataFrame with
        one row per dataset and ``figs`` is a dict of bar-plot figures keyed by
        metric name, or ``None`` if plotting fails.
    """
    real_df  = pd.read_csv(real_csv)
    synth_df = pd.read_csv(synth_csv)

    real_df.columns  = real_df.columns.str.strip()
    synth_df.columns = synth_df.columns.str.strip()

    if required_fields is None:
        required_fields = list(
            set(real_df.columns).intersection(set(synth_df.columns))
        )

    real_comp  = compute_completeness(real_df,  required_fields=required_fields, label=f"{label}_real")
    synth_comp = compute_completeness(synth_df, required_fields=required_fields, label=f"{label}_synth")

    real_per_field  = real_comp.pop('per_field',  {})
    synth_per_field = synth_comp.pop('per_field', {})

    comp_df = pd.DataFrame([
        {'Dataset': 'Real',  **real_comp},
        {'Dataset': 'Synth', **synth_comp}
    ])

    try:
        from src.visualization import create_barplot

        figs = {}

        df_melt = comp_df.melt(
            id_vars="Dataset",
            value_vars=["Missing_Data_Percentage", "Required_Fields_Completeness"],
            var_name="Metric",
            value_name="Value"
        )
        summary_metrics = ["Missing_Data_Percentage", "Required_Fields_Completeness"]
        if metrics_to_include:
            summary_metrics = [
                m for m in summary_metrics
                if metrics_to_include.get(m, True)
            ]

        for metric in summary_metrics:
            sub = df_melt[df_melt["Metric"] == metric]
            pretty = metric.replace("_", " ").title()
            figs[metric] = create_barplot(
                sub,
                x="Value",
                y="Dataset",
                hue="Dataset",
                suptitle=f"Completeness — {pretty}",
                xlabel="Percentage (%)",
                bar_label_fmt="%.2f",
            )

        include_per_field = True
        if metrics_to_include:
            include_per_field = metrics_to_include.get("Per_Field", True)

        if include_per_field and (real_per_field or synth_per_field):
            all_fields = sorted(
                set(list(real_per_field.keys()) + list(synth_per_field.keys()))
            )
            per_rows = []
            for field in all_fields:
                per_rows.append({"Dataset": "Real", "Field": field,
                                 "Completeness (%)": real_per_field.get(field, float("nan"))})
                per_rows.append({"Dataset": "Synth", "Field": field,
                                 "Completeness (%)": synth_per_field.get(field, float("nan"))})
            per_df = pd.DataFrame(per_rows)
            figs["Per_Field"] = create_barplot(
                per_df,
                x="Completeness (%)",
                y="Field",
                hue="Dataset",
                suptitle="Per-Field Completeness",
                xlabel="Completeness (%)",
                bar_label_fmt="%.1f",
                height_per_category=0.45,
            )

    except Exception:
        figs = None

    return comp_df, figs


def consistency_analysis(
    real_csv,
    synth_csv,
    group_by="Race",
    metric_cols=None,
    label="",
    metrics_to_plot=None,
):
    """Evaluate numeric metadata consistency across demographic subgroups.

    Args:
        real_csv: Path to the real-data metadata CSV.
        synth_csv: Path to the synthetic-data metadata CSV.
        group_by: Categorical column used to partition rows before computing
            consistency statistics.
        metric_cols: Numeric column names to analyze for within-group stability.
        label: Prefix applied to internal consistency computation labels.
        metrics_to_plot: Optional mapping of consistency metric name to bool;
            only metrics marked ``True`` are plotted.

    Returns:
        A tuple ``(cons_df, figs)`` where ``cons_df`` holds per-dataset,
        per-metric consistency results and ``figs`` is a dict of bar-plot figures
        keyed by consistency metric name, or ``None`` if plotting fails. Returns
        ``(empty DataFrame, None)`` when no results are produced.
    """
    real_df  = pd.read_csv(real_csv)
    synth_df = pd.read_csv(synth_csv)

    real_df.columns  = real_df.columns.str.strip()
    synth_df.columns = synth_df.columns.str.strip()

    for col_name, df in [("real_csv", real_df), ("synth_csv", synth_df)]:
        if group_by not in df.columns:
            raise ValueError(f"group_by column '{group_by}' not found in {col_name}.")

    real_df  = real_df.replace(r'^\s*$', np.nan, regex=True).dropna(subset=[group_by])
    synth_df = synth_df.replace(r'^\s*$', np.nan, regex=True).dropna(subset=[group_by])

    if not metric_cols:
        raise ValueError("No valid metric_cols found in both CSVs.")

    results = []

    for metric in metric_cols:
        if metric in real_df.columns:
            real_group_data = {
                str(g): group[metric].dropna().values
                for g, group in real_df.groupby(group_by)
                if len(group[metric].dropna()) > 0
            }
            if len(real_group_data) >= 2:
                real_cons = compute_consistency(real_group_data, label=f"{label}_real_{metric}")
                results.append({'Dataset': 'Real', 'Group_By': group_by, 'Metric': metric, **real_cons})

        if metric in synth_df.columns:
            synth_group_data = {
                str(g): group[metric].dropna().values
                for g, group in synth_df.groupby(group_by)
                if len(group[metric].dropna()) > 0
            }
            if len(synth_group_data) >= 2:
                synth_cons = compute_consistency(synth_group_data, label=f"{label}_synth_{metric}")
                results.append({'Dataset': 'Synth', 'Group_By': group_by, 'Metric': metric, **synth_cons})

    if not results:
        return pd.DataFrame(), None

    cons_df = pd.DataFrame(results)

    try:
        from src.visualization import create_barplot
        plot_metrics = ["Variance_of_Group_Means", "Max_Min_Difference", "ANOVA_F_statistic"]
        if metrics_to_plot:
            plot_metrics = [m for m in plot_metrics if metrics_to_plot.get(m, True)]
        id_vars = ["Metric", "Dataset"] if "Dataset" in cons_df.columns else ["Metric"]
        df_melt = cons_df.melt(
            id_vars=id_vars,
            value_vars=plot_metrics,
            var_name="Consistency_Metric",
            value_name="Value"
        )

        figs = {}
        for cmetric in df_melt["Consistency_Metric"].dropna().unique():
            sub = df_melt[df_melt["Consistency_Metric"] == cmetric]
            pretty = cmetric.replace("_", " ").title()
            figs[cmetric] = create_barplot(
                sub,
                x="Value",
                y="Metric",
                hue="Dataset" if "Dataset" in sub.columns else None,
                suptitle=f"Consistency — {pretty} (by {group_by})",
                bar_label_fmt="%.4f",
            )
    except Exception:
        figs = None

    return cons_df, figs

def _resolve_image_path(path, path_root=None):
    """Resolve a CSV filepath, falling back to path_root for repo-root-relative paths."""
    path = str(path)

    if os.path.isfile(path):
        return path

    if path_root:
        candidate = os.path.normpath(os.path.join(path_root, path))

        if os.path.isfile(candidate):
            return candidate

    return None


def _sample_image_pairs(real_df, synth_df, n=5, seed=None, path_root=None):
    """
    Sample up to n (real, synth) image path pairs, matching each real image to a
    synthetic image whose filename contains the real filename stem — the same
    pairing used by the webapp preview.
    """
    if 'filepath' not in real_df.columns or 'filepath' not in synth_df.columns:
        return []

    rng         = np.random.default_rng(seed)
    order       = rng.permutation(len(real_df))
    synth_paths = synth_df['filepath'].astype(str)
    pairs       = []

    for idx in order:
        if len(pairs) >= n:
            break

        real_path = _resolve_image_path(real_df.iloc[idx]['filepath'], path_root)

        if real_path is None:
            continue

        real_stem = os.path.splitext(os.path.basename(real_path))[0]
        matching  = synth_df[synth_paths.str.contains(real_stem, regex=False, na=False)]

        if matching.empty:
            continue

        synth_row  = matching.sample(1, random_state=int(rng.integers(0, 2**31))).iloc[0]
        synth_path = _resolve_image_path(synth_row['filepath'], path_root)

        if synth_path is None:
            continue

        pairs.append((real_path, synth_path))

    return pairs


def scorecard_analysis(
    real_csv,
    synth_csv,
    metadata_real_csv=None,
    metadata_synth_csv=None,
    results_dir='./data/features',
    real_features='real_patch_appearance_features.npz',
    synth_features='kde_patch_appearance_features.npz',
    features_to_check=None,
    n_images=5,
    seed=None,
    path_root=None,
):
    """Build a summary scorecard comparing real and synthetic datasets.

    Aggregates sample counts, metadata completeness, feature-space coverage
    variance, constraint violations, and paired image previews into a single
    dashboard figure.

    Args:
        real_csv: Path to the real-data image or metadata CSV (used for sample
            counts and image pairing).
        synth_csv: Path to the synthetic-data image or metadata CSV.
        metadata_real_csv: Optional separate real metadata CSV for completeness;
            defaults to ``real_csv``.
        metadata_synth_csv: Optional separate synthetic metadata CSV for
            completeness; defaults to ``synth_csv``.
        results_dir: Directory containing patch feature NPZ files.
        real_features: Filename of the real-data feature NPZ within ``results_dir``.
        synth_features: Filename of the synthetic-data feature NPZ within
            ``results_dir``.
        features_to_check: Optional subset of patch features for constraint
            violation analysis.
        n_images: Maximum number of real/synthetic image pairs to display.
        seed: Optional random seed for reproducible image-pair sampling.
        path_root: Optional base directory for resolving relative image paths.

    Returns:
        A tuple ``(summary_df, fig)`` where ``summary_df`` compares datasets on
        key metrics and ``fig`` is the composite scorecard figure.
    """
    from src.visualization import create_scorecard_figure

    real_df               = pd.read_csv(real_csv)
    synth_df              = pd.read_csv(synth_csv)

    n_real                = len(real_df)
    n_synth               = len(synth_df)

    meta_real_df          = pd.read_csv(metadata_real_csv or real_csv)
    meta_synth_df         = pd.read_csv(metadata_synth_csv or synth_csv)

    meta_real_df.columns  = meta_real_df.columns.str.strip()
    meta_synth_df.columns = meta_synth_df.columns.str.strip()

    required_fields = list(
        set(meta_real_df.columns).intersection(set(meta_synth_df.columns))
    )

    real_comp    = compute_completeness(meta_real_df,  required_fields=required_fields, label="scorecard_real")
    synth_comp   = compute_completeness(meta_synth_df, required_fields=required_fields, label="scorecard_synth")

    completeness = {
        'Real' : real_comp.get('Required_Fields_Completeness'),
        'Synth': synth_comp.get('Required_Fields_Completeness'),
    }

    real_features_path  = os.path.join(results_dir, real_features)
    synth_features_path = os.path.join(results_dir, synth_features)

    real_feat_df        = combine_features([real_features_path])
    synth_feat_df       = combine_features([synth_features_path])

    real_feat_df, synth_feat_df, kept_features = filter_features(real_feat_df, synth_feat_df, 0.5)

    coverage = {
        'Real' : compute_coverage(real_feat_df[kept_features],  "Real").get('Variance'),
        'Synth': compute_coverage(synth_feat_df[kept_features], "Synth").get('Variance'),
    }

    violation_df = compute_constraint(
        load_features(real_features_path),
        load_features(synth_features_path),
        features_to_check=features_to_check,
    )

    image_pairs = _sample_image_pairs(real_df, synth_df, n=n_images, seed=seed, path_root=path_root)

    summary_rows = [
        ('Real samples',      n_real),
        ('Synthetic samples', n_synth),
        ('Image features',    len(kept_features)),
        ('Metadata fields',   len(required_fields)),
    ]

    summary_df = pd.DataFrame([
        {
            'Dataset'                     : 'Real',
            'Samples'                     : n_real,
            'Required_Fields_Completeness': completeness['Real'],
            'Coverage_Variance'           : coverage['Real'],
        },
        {
            'Dataset'                     : 'Synth',
            'Samples'                     : n_synth,
            'Required_Fields_Completeness': completeness['Synth'],
            'Coverage_Variance'           : coverage['Synth'],
        },
    ])

    fig = create_scorecard_figure(
        summary_rows,
        completeness,
        coverage,
        violation_df,
        image_pairs,
    )

    return summary_df, fig


def constraint_patch_analysis(
    results_dir='./data/features',
    real_features='real_patch_appearance_features.npz',
    synth_features='kde_patch_appearance_features.npz',
    features_to_check=None,
):
    """Run constraint analysis on patch features loaded from NPZ files.

    Args:
        results_dir: Directory containing patch feature NPZ files.
        real_features: Filename of the real-data feature NPZ within ``results_dir``.
        synth_features: Filename of the synthetic-data feature NPZ within
            ``results_dir``.
        features_to_check: Optional subset of feature names to evaluate for
            percentile-bound violations.

    Returns:
        The result of ``constraint_analysis`` applied to the loaded feature dicts.
    """
    real_feats = load_features(os.path.join(results_dir, real_features))
    synth_feats = load_features(os.path.join(results_dir, synth_features))
    return constraint_analysis(real_feats, synth_feats, features_to_check=features_to_check)


def constraint_analysis(real_features, synth_features, features_to_check=None):
    """Compute constraint violations and plot synthetic violation rates by feature.

    Args:
        real_features: Mapping of feature name to real-data value arrays.
        synth_features: Mapping of feature name to synthetic-data value arrays.
        features_to_check: Optional subset of feature names to evaluate. When
            omitted, all shared features are checked.

    Returns:
        A tuple ``(violation_df, fig)`` where ``violation_df`` reports per-feature
        violation statistics and ``fig`` is a range plot showing each feature's
        synthetic spread against its real-data allowed range.
    """
    from src.visualization import create_constraint_range_plot

    violation_df = compute_constraint(
        real_features,
        synth_features,
        features_to_check = features_to_check,
    )

    fig = create_constraint_range_plot(
        violation_df if violation_df is not None else pd.DataFrame(),
    )

    return violation_df, fig