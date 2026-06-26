import numpy              as     np
import pandas             as     pd
import seaborn            as     sns

from   src.routines       import bootstrap_ci
from   scipy.stats        import gaussian_kde
from   skimage            import color, filters, measure
from   skimage.transform  import resize
from   skimage.morphology import skeletonize
from   scipy.stats        import gaussian_kde

from   matplotlib         import pyplot as plt

PLOT_DPI = 220

def apply_large_plot_style():
    sns.set_theme(style="whitegrid", context="talk")

    plt.rcParams.update({
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,

        "font.size": 15,
        "axes.titlesize": 20,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,

        "axes.titleweight": "bold",
        "axes.labelpad": 10,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,

        "figure.titlesize": 28,
        "figure.titleweight": "bold",
    })

def print_histograms(real_features, synth_features):
    apply_large_plot_style()

    feature_names = list(real_features.keys())
    n             = len(feature_names)

    fig_height    = max(7 * n, 8)
    fig, axes     = plt.subplots(
        n,
        1,
        figsize=(17, fig_height),
        dpi=PLOT_DPI,
        sharex=False
    )

    if n == 1:
        axes = np.atleast_1d(axes)

    for ax, feature_name in zip(axes, feature_names):
        real_data  = np.asarray(real_features[feature_name], dtype=float)
        synth_data = np.asarray(synth_features[feature_name], dtype=float)

        real_clean  = real_data[~np.isnan(real_data)]
        synth_clean = synth_data[~np.isnan(synth_data)]

        real_mean  = np.nanmean(real_clean)
        real_ci    = bootstrap_ci(real_clean)

        synth_mean = np.nanmean(synth_clean)
        synth_ci   = bootstrap_ci(synth_clean)

        x_min = np.nanmin([np.nanmin(real_clean), np.nanmin(synth_clean)])
        x_max = np.nanmax([np.nanmax(real_clean), np.nanmax(synth_clean)])

        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        shared_range = (x_min, x_max)

        ax.hist(
            real_clean,
            bins=35,
            alpha=0.6,
            color="#9999CC",
            label=f"Real (n={len(real_clean)})",
            density=True,
            range=shared_range,
            edgecolor="black",
            linewidth=0.6
        )

        ax.hist(
            synth_clean,
            bins=35,
            alpha=0.6,
            color="#FF9966",
            label=f"Synthetic (n={len(synth_clean)})",
            density=True,
            range=shared_range,
            edgecolor="black",
            linewidth=0.6
        )

        x_vals = np.linspace(x_min, x_max, 400)

        if len(np.unique(real_clean)) > 1:
            ax.plot(
                x_vals,
                gaussian_kde(real_clean)(x_vals),
                color="#9999CC",
                linewidth=3
            )

        if len(np.unique(synth_clean)) > 1:
            ax.plot(
                x_vals,
                gaussian_kde(synth_clean)(x_vals),
                color="#FF9966",
                linewidth=3
            )

        ax.axvline(real_mean, color="#9999CC", linestyle="--", linewidth=3, alpha=0.85)
        ax.axvline(synth_mean, color="#FF9966", linestyle="--", linewidth=3, alpha=0.85)

        ax.axvspan(real_ci[0], real_ci[1], alpha=0.12, color="#9999CC", label="Real 95% CI")
        ax.axvspan(synth_ci[0], synth_ci[1], alpha=0.12, color="#FF9966", label="Synthetic 95% CI")

        pretty_name = feature_name.replace("_", " ").title()

        ax.set_title(
            f"Real vs. Synthetic {pretty_name} Distribution",
            fontsize=21,
            fontweight="bold",
            pad=14
        )
        ax.set_xlabel(pretty_name, fontsize=17, labelpad=10)
        ax.set_ylabel("Density", fontsize=17, labelpad=10)
        ax.set_xlim(shared_range)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(loc="best", fontsize=13, frameon=True)

    fig.suptitle(
        "Real vs Synthetic Feature Distributions",
        fontsize=28,
        fontweight="bold",
        y=0.995
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=3.0)

    return fig


def visualize_gabor_and_skeleton_examples(image, mask, n_examples=4):
    """
    Visualizes Gabor filter responses and skeletons for a sample of nuclei.

    Parameters
    ----------
    image    : np.ndarray  - grayscale or RGB image
    mask     : np.ndarray  - integer label mask (each nucleus has a unique label)
    n_examples : int       - number of nuclei to visualize
    """

    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    if gray.shape != mask.shape:
        mask = resize(
            mask, gray.shape,
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(mask.dtype)

    gabor_frequencies  = [0.1, 0.3, 0.5]
    gabor_orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    gabor_responses    = {}

    for freq in gabor_frequencies:
        for theta in gabor_orientations:
            real, imag = filters.gabor(gray, frequency=freq, theta=theta)
            gabor_responses[(freq, theta)] = np.sqrt(real**2 + imag**2)

    props   = measure.regionprops(mask, intensity_image=gray)
    props   = [p for p in props if p.image.shape[0] > 5 and p.image.shape[1] > 5]
    sampled = props[:n_examples]

    for i, prop in enumerate(sampled):
        minr, minc, maxr, maxc        = prop.bbox
        nucleus_region                = gray[minr:maxr, minc:maxc]
        nucleus_mask                  = prop.image
        skeleton                      = skeletonize(nucleus_mask)

        display_region                = nucleus_region.copy()
        display_region[~nucleus_mask] = 0

        fig, axes                     = plt.subplots(1, 3, figsize=(10, 3))
        fig.suptitle(f"Nucleus {i+1} — Skeleton", fontsize=13)

        axes[0].imshow(display_region, cmap='gray')
        axes[0].set_title("Grayscale crop")
        axes[0].axis('off')

        axes[1].imshow(nucleus_mask, cmap='gray')
        axes[1].set_title("Binary mask")
        axes[1].axis('off')

        axes[2].imshow(nucleus_mask, cmap='gray', alpha=0.5)
        axes[2].imshow(skeleton, cmap='hot', alpha=0.8)
        axes[2].set_title(f"Skeleton (length={int(np.sum(skeleton))})")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, len(gabor_frequencies) + 1, figsize=(14, 3))
        fig.suptitle(f"Nucleus {i+1} — Gabor Responses (mean over orientations)", fontsize=13)

        axes[0].imshow(display_region, cmap='gray')
        axes[0].set_title("Grayscale crop")
        axes[0].axis('off')

        for j, freq in enumerate(gabor_frequencies):
            stacked = np.stack([
                gabor_responses[(freq, theta)][minr:maxr, minc:maxc]
                for theta in gabor_orientations
            ], axis=0)
            mean_response = np.mean(stacked, axis=0)
            mean_response[~nucleus_mask] = 0

            axes[j+1].imshow(mean_response, cmap='inferno')
            axes[j+1].set_title(f"freq={freq}")
            axes[j+1].axis('off')

        plt.tight_layout()
        plt.show()

def create_coverage_barplot(coverage_df: pd.DataFrame):
    apply_large_plot_style()

    metrics = [
        "Variance",
        "Entropy",
        "Distance_to_Centroid",
        "Convex_Hull_Volume"
    ]

    df_plot = coverage_df.reset_index().rename(columns={"index": "Dataset"})
    df_melt = df_plot.melt(
        id_vars    = "Dataset",
        var_name   = "Metric",
        value_name = "Value"
    )

    n = len(metrics)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(16, 6.2 * n),
        dpi=PLOT_DPI
    )

    if n == 1:
        axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics):
        data = df_melt[df_melt["Metric"] == metric]

        sns.barplot(
            data=data,
            x="Dataset",
            y="Value",
            ax=ax,
            palette="Set2",
            edgecolor="black",
            linewidth=0.9
        )

        ax.set_title(
            metric.replace("_", " "),
            fontsize=21,
            fontweight="bold",
            pad=14
        )
        ax.set_xlabel("")
        ax.set_ylabel("Value",   fontsize  = 17,  labelpad= 10)
        ax.tick_params(axis="x", labelsize = 15,  rotation= 0)
        ax.tick_params(axis="y", labelsize = 14)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f", padding=5, fontsize=13)

        ax.margins(y=0.18)

    fig.suptitle(
        "Real vs Synthetic Coverage",
        fontsize   = 28,
        fontweight = "bold",
        y          = 0.995
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=3.0)

    return fig

def create_congruence_barplot(congruence_df: pd.DataFrame, metrics_to_compute):
    apply_large_plot_style()

    metrics = list(metrics_to_compute)

    df_melt = congruence_df.melt(
        id_vars    = ["Feature"],
        value_vars = metrics,
        var_name   = "Metric",
        value_name = "Value"
    )

    n_features = congruence_df["Feature"].nunique()
    fig_height = max(20, 1.2 * n_features + 9)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, fig_height),
        dpi=PLOT_DPI
    )

    for ax, metric in zip(axes, metrics):
        data = (
            df_melt[df_melt["Metric"] == metric]
            .sort_values("Value", ascending=True)
        )

        sns.barplot(
            data=data,
            y="Feature",
            x="Value",
            ax=ax,
            palette="viridis",
            edgecolor="black",
            linewidth=0.6
        )

        ax.set_title(
            metric.replace("_", " "),
            fontsize=22,
            fontweight="bold",
            pad=14
        )
        ax.set_ylabel("")
        ax.set_xlabel("Value", fontsize=17, labelpad=10)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        "Real vs. Synthetic Statistical Congruence",
        fontsize=30,
        fontweight="bold",
        y=0.995
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=3.2)

    return fig

def create_completeness_barplot(
    comp_df: pd.DataFrame,
    real_per_field: dict = None,
    synth_per_field: dict = None
):
    apply_large_plot_style()

    metrics = [
        "Missing_Data_Percentage",
        "Required_Fields_Completeness"
    ]

    df_melt = comp_df.melt(
        id_vars="Dataset",
        value_vars=metrics,
        var_name="Metric",
        value_name="Value"
    )

    has_per_field = bool(real_per_field or synth_per_field)

    if not has_per_field:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(17, 7),
            dpi=PLOT_DPI
        )

        for ax, metric in zip(axes, metrics):
            data = df_melt[df_melt["Metric"] == metric]

            sns.barplot(
                data=data,
                x="Dataset",
                y="Value",
                ax=ax,
                palette="Set2",
                edgecolor="black",
                linewidth=0.9
            )

            ax.set_title(
                metric.replace("_", " "),
                fontsize=21,
                fontweight="bold",
                pad=14
            )
            ax.set_xlabel("")
            ax.set_ylabel("Percentage (%)", fontsize=17, labelpad=10)
            ax.set_ylim(0, 115)
            ax.tick_params(axis="both", labelsize=14)

            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f%%", padding=5, fontsize=13)

        fig.suptitle(
            "Completeness: Real vs Synthetic",
            fontsize=28,
            fontweight="bold",
            y=0.995
        )

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    all_fields = sorted(
        set(list(real_per_field.keys()) + list(synth_per_field.keys()))
    )

    per_field_rows = []

    for field in all_fields:
        per_field_rows.append({
            "Dataset": "Real",
            "Field": field,
            "Completeness (%)": real_per_field.get(field, np.nan)
        })
        per_field_rows.append({
            "Dataset": "Synth",
            "Field": field,
            "Completeness (%)": synth_per_field.get(field, np.nan)
        })

    per_field_df = pd.DataFrame(per_field_rows)

    n_fields = len(all_fields)
    fig_height = max(14, 0.55 * n_fields + 9)

    fig = plt.figure(figsize=(18, fig_height), dpi=PLOT_DPI)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, max(1.7, 0.12 * n_fields + 1.4)]
    )

    top_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1])
    ]

    ax_per = fig.add_subplot(gs[1, :])

    for ax, metric in zip(top_axes, metrics):
        data = df_melt[df_melt["Metric"] == metric]

        sns.barplot(
            data=data,
            x="Dataset",
            y="Value",
            ax=ax,
            palette="Set2",
            edgecolor="black",
            linewidth=0.9
        )

        ax.set_title(
            metric.replace("_", " "),
            fontsize=21,
            fontweight="bold",
            pad=14
        )
        ax.set_xlabel("")
        ax.set_ylabel("Percentage (%)", fontsize=17, labelpad=10)
        ax.set_ylim(0, 115)
        ax.tick_params(axis="both", labelsize=14)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f%%", padding=5, fontsize=13)

    sns.barplot(
        data=per_field_df,
        y="Field",
        x="Completeness (%)",
        hue="Dataset",
        ax=ax_per,
        palette="Set2",
        edgecolor="black",
        linewidth=0.8
    )

    ax_per.set_title(
        "Per-Field Completeness: Real vs Synthetic",
        fontsize=22,
        fontweight="bold",
        pad=16
    )
    ax_per.set_xlabel("Completeness (%)", fontsize=17, labelpad=10)
    ax_per.set_ylabel("Metadata Field", fontsize=17, labelpad=10)
    ax_per.set_xlim(0, 115)
    ax_per.tick_params(axis="y", labelsize=14)
    ax_per.tick_params(axis="x", labelsize=14)
    ax_per.legend(title="Dataset", fontsize=13, title_fontsize=14)

    for container in ax_per.containers:
        ax_per.bar_label(container, fmt="%.1f%%", padding=4, fontsize=12)

    fig.suptitle(
        "Completeness: Real vs Synthetic",
        fontsize=30,
        fontweight="bold",
        y=0.995
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=3.0)

    return fig

def create_consistency_barplot(cons_df: pd.DataFrame, group_by):
    apply_large_plot_style()

    if cons_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5), dpi=PLOT_DPI)
        ax.text(
            0.5, 0.5,
            "No consistency data available",
            ha="center", va="center", fontsize=18
        )
        ax.axis("off")
        return fig

    metrics_to_plot = [
        "Variance_of_Group_Means",
        "Max_Min_Difference",
        "ANOVA_F_statistic"
    ]

    id_vars = ["Metric", "Dataset"] if "Dataset" in cons_df.columns else ["Metric"]

    df_melt = cons_df.melt(
        id_vars=id_vars,
        value_vars=metrics_to_plot,
        var_name="Consistency_Metric",
        value_name="Value"
    )

    n_metrics  = cons_df["Metric"].nunique()
    fig_height = max(20, 1.15 * n_metrics + 9)

    fig, axes = plt.subplots(3, 1, figsize=(18, fig_height), dpi=PLOT_DPI)

    for ax, cmetric in zip(axes, metrics_to_plot):
        data = df_melt[df_melt["Consistency_Metric"] == cmetric]

        sns.barplot(
            data=data,
            y="Metric",
            x="Value",
            hue="Dataset" if "Dataset" in data.columns else None,
            ax=ax,
            palette="Set2",
            edgecolor="black",
            linewidth=0.7
        )

        ax.set_title(
            cmetric.replace("_", " "),
            fontsize=22, fontweight="bold", pad=14
        )
        ax.set_xlabel("Value", fontsize=17, labelpad=10)
        ax.set_ylabel("", fontsize=17)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(True, axis="x", alpha=0.3)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f", padding=5, fontsize=12)

        ax.margins(x=0.15)

        if "Dataset" in data.columns:
            ax.legend(title="Dataset", fontsize=13, title_fontsize=14)

    fig.suptitle(
        f"Consistency across {group_by}: Real vs. Synthetic",
        fontsize=30, fontweight="bold", y=0.995
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=3.2)

    return fig
