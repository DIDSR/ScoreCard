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

def hist_analysis(results_dir="./data/features", feature_names=None):
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
    real_features                    = os.path.join(results_dir, real_features)
    synth_features                   = os.path.join(results_dir, synth_features)

    real_features_list               = []
    synth_features_list              = []

    real_features_list.append(real_features)
    synth_features_list.append(synth_features)

    real_df                          = combine_features(real_features_list)
    synth_df                         = combine_features(synth_features_list)

    real_df, synth_df, kept_features = filter_features(real_df, synth_df, 0.5)
    cols = [c for c in kept_features if c not in real_features]

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

    real_features        = os.path.join(results_dir, real_features)
    synth_features       = os.path.join(results_dir, synth_features)

    real_features_list   = []
    synth_features_list  = []

    real_features_list.append(real_features)
    synth_features_list.append(synth_features)

    real_df              = combine_features(real_features_list)
    synth_df             = combine_features(synth_features_list)

    congruence_results = {}
    cols = real_df.columns.tolist()
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

def completeness_analysis(real_csv, synth_csv, required_fields=None, label="", metrics_to_include=None):
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
    real_df  = pd.read_csv(real_csv)
    synth_df = pd.read_csv(synth_csv)

    real_df.columns  = real_df.columns.str.strip()
    synth_df.columns = synth_df.columns.str.strip()

    for col_name, df in [("real_csv", real_df), ("synth_csv", synth_df)]:
        if group_by not in df.columns:
            raise ValueError(f"group_by column '{group_by}' not found in {col_name}.")

    real_df  = real_df.replace(r'^\s*$', np.nan, regex=True).dropna(subset=[group_by])
    synth_df = synth_df.replace(r'^\s*$', np.nan, regex=True).dropna(subset=[group_by])

    if metric_cols is None:
        default_metrics = [
            'Age at dx',
            'BMI at dx (kg)',
            'BMI at follow-up (kg)',
            'mpp',
            'compressionratio',
            'exposure time'
        ]
        metric_cols = [
            c for c in default_metrics
            if c in real_df.columns and c in synth_df.columns
        ]

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

def constraint_patch_analysis(
    results_dir='./data/features',
    real_features='real_patch_appearance_features.npz',
    synth_features='kde_patch_appearance_features.npz',
    features_to_check=None,
):
    real_feats = load_features(os.path.join(results_dir, real_features))
    synth_feats = load_features(os.path.join(results_dir, synth_features))
    return constraint_analysis(real_feats, synth_feats, features_to_check=features_to_check)


def constraint_analysis(real_features, synth_features, features_to_check=None):
    from src.visualization import create_barplot

    violation_df = compute_constraint(
        real_features,
        synth_features,
        features_to_check=features_to_check,
    )

    fig = create_barplot(
        violation_df if violation_df is not None else pd.DataFrame(),
        x="Synth_Violation_%",
        y="Feature",
        suptitle="Constraint Violation Rate in Synthetic Data",
        xlabel="Violation Rate (%)",
        bar_label_fmt="%.1f",
        color="#EE6666",
        sort=False,
        height_per_category=0.55,
        base_width=14.0,
    )

    return violation_df, fig