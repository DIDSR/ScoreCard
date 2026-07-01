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
WEB_FIG_WIDTH = 12.5
WEB_SUPTITLE_SIZE = 20

def _finalize_figure(fig, *, top=0.93):
    fig.tight_layout(rect=[0, 0, 1, top], pad=1.4, h_pad=1.2, w_pad=1.0)

    return fig

def apply_large_plot_style():
    sns.set_theme(style="whitegrid", context="talk")

    plt.rcParams.update({
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,

        "font.size"         : 15,
        "axes.titlesize"    : 20,
        "axes.labelsize"    : 17,
        "xtick.labelsize"   : 14,
        "ytick.labelsize"   : 14,
        "legend.fontsize"   : 13,

        "axes.titleweight"  : "bold",
        "axes.labelpad"     : 10,
        "xtick.major.pad"   : 6,
        "ytick.major.pad"   : 6,

        "figure.titlesize"  : 28,
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
        figsize =(17, fig_height),
        dpi     =PLOT_DPI,
        sharex =False
    )

    if n == 1:
        axes = np.atleast_1d(axes)

    for ax, feature_name in zip(axes, feature_names):
        real_data   = np.asarray(real_features[feature_name], dtype=float)
        synth_data  = np.asarray(synth_features[feature_name], dtype=float)

        real_clean  = real_data[~np.isnan(real_data)]
        synth_clean = synth_data[~np.isnan(synth_data)]

        real_mean   = np.nanmean(real_clean)
        real_ci     = bootstrap_ci(real_clean)

        synth_mean  = np.nanmean(synth_clean)
        synth_ci    = bootstrap_ci(synth_clean)

        x_min       = np.nanmin([np.nanmin(real_clean), np.nanmin(synth_clean)])
        x_max       = np.nanmax([np.nanmax(real_clean), np.nanmax(synth_clean)])

        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        shared_range = (x_min, x_max)

        ax.hist(
            real_clean,
            bins      = 35,
            alpha     = 0.6,
            color     = "#9999CC",
            label     = f"Real (n={len(real_clean)})",
            density   = True,
            range     = shared_range,
            edgecolor = "black",
            linewidth = 0.6
        )

        ax.hist(
            synth_clean,
            bins      = 35,
            alpha     = 0.6,
            color     = "#FF9966",
            label     = f"Synthetic (n={len(synth_clean)})",
            density   = True,
            range     = shared_range,
            edgecolor = "black",
            linewidth = 0.6
        )

        x_vals = np.linspace(x_min, x_max, 400)

        if len(np.unique(real_clean)) > 1:
            ax.plot(
                x_vals,
                gaussian_kde(real_clean)(x_vals),
                color     = "#9999CC",
                linewidth = 3
            )

        if len(np.unique(synth_clean)) > 1:
            ax.plot(
                x_vals,
                gaussian_kde(synth_clean)(x_vals),
                color     = "#FF9966",
                linewidth = 3
            )

        ax.axvline(real_mean,  color="#9999CC", linestyle="--", linewidth=3, alpha=0.85)
        ax.axvline(synth_mean, color="#FF9966", linestyle="--", linewidth=3, alpha=0.85)

        ax.axvspan(real_ci[0],  real_ci[1],  alpha=0.12, color="#9999CC", label="Real 95% CI")
        ax.axvspan(synth_ci[0], synth_ci[1], alpha=0.12, color="#FF9966", label="Synthetic 95% CI")

        pretty_name = feature_name.replace("_", " ").title()

        ax.set_title(
            f"Real vs. Synthetic {pretty_name} Distribution",
            fontsize   = 21,
            fontweight = "bold",
            pad        = 14
        )
        ax.set_xlabel(pretty_name, fontsize=17, labelpad=10)
        ax.set_ylabel("Density", fontsize=17, labelpad=10)
        ax.set_xlim(shared_range)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(loc="best", fontsize=13, frameon=True)

    fig.suptitle(
        "Real vs Synthetic Feature Distributions",
        fontsize   = 28,
        fontweight = "bold",
        y          = 0.995
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
            mask, 
            gray.shape,
            order          = 0, 
            preserve_range = True, 
            anti_aliasing  = False
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

            mean_response                = np.mean(stacked, axis=0)
            mean_response[~nucleus_mask] = 0

            axes[j+1].imshow(mean_response, cmap='inferno')
            axes[j+1].set_title(f"freq={freq}")
            axes[j+1].axis('off')

        plt.tight_layout()
        plt.show()

def create_barplot(
    df                 : pd.DataFrame,
    *,
    x                  : str = "Value",
    y                  : str = "Category",
    hue                : str | None = None,
    facet              : str | None = None,
    suptitle           : str | None = None,
    xlabel             : str | None = None,
    ylabel             : str | None = None,
    sort               : bool  = True,
    ascending          : bool  = True,
    bar_label_fmt      : str   = "%.4f",
    color              : str   = "#5C6BC0",
    height_per_category: float = 0.58,
    base_width         : float = WEB_FIG_WIDTH,
) -> "plt.Figure | dict[str, plt.Figure]":

    apply_large_plot_style()

    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(12, 5), dpi=PLOT_DPI)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=18)
        ax.axis("off")

        return fig

    working     = df.copy()
    hue_palette = ["#5470C6", "#EE6666"]

    if facet and facet in working.columns:
        facet_values                = [v for v in working[facet].dropna().unique()]
        figs: dict[str, plt.Figure] = {}

        for fval in facet_values:
            sub = working[working[facet] == fval].copy()

            if sort and x in sub.columns and len(sub) > 1:
                sub = sub.sort_values(by=x, ascending=ascending)

            n_cats  = sub[y].nunique() if y in sub.columns else 6
            h       = max(3.8, height_per_category * n_cats + 2.0)

            fig, ax = plt.subplots(figsize=(base_width, h), dpi=PLOT_DPI)

            _plot_bars_on_ax(
                ax, 
                sub, 
                x, 
                y, 
                hue, 
                color, 
                hue_palette, 
                bar_label_fmt,
                xlabel = xlabel, 
                ylabel = ylabel
            )

            facet_title = str(fval).replace("_", " ").title()

            if suptitle:
                fig.suptitle(
                    f"{suptitle} — {facet_title}",
                    fontsize=WEB_SUPTITLE_SIZE,
                    fontweight="bold",
                    y=0.98,
                )
            else:
                ax.set_title(facet_title, fontsize=18, fontweight="bold", pad=10)

            _finalize_figure(fig)
            figs[str(fval)] = fig

        return figs

    n_cats  = working[y].nunique() if y in working.columns else 6
    h       = max(3.8, height_per_category * n_cats + 2.0)
    fig, ax = plt.subplots(figsize=(base_width, h), dpi=PLOT_DPI)

    _plot_bars_on_ax(
        ax, working, 
        x, 
        y, 
        hue, 
        color, 
        hue_palette, 
        bar_label_fmt,
        xlabel = xlabel, 
        ylabel = ylabel
    )

    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize   = WEB_SUPTITLE_SIZE,
            fontweight = "bold",
            y          = 0.98,
        )

    _finalize_figure(fig)
    return fig


def _plot_bars_on_ax(ax, sub, x, y, hue, color, hue_palette, bar_label_fmt, xlabel=None, ylabel=None):
    plot_kw = dict(
        data      = sub,
        y         = y,
        x         = x,
        ax        = ax,
        edgecolor = "black",
        linewidth = 0.75,
    )

    if hue and hue in sub.columns:
        plot_kw["hue"]     = hue
        plot_kw["palette"] = hue_palette
    else:
        plot_kw["color"]   = color

    sns.barplot(**plot_kw)

    ax.set_xlabel(
        xlabel or (x.replace("_", " ").title() if x else ""),
        fontsize = 16, 
        labelpad = 8
    )
    ax.set_ylabel(ylabel or "", fontsize=16, labelpad=6)

    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(True, axis="x", alpha=0.28)

    for container in ax.containers:
        ax.bar_label(container, fmt=bar_label_fmt, padding=4, fontsize=11)

    ax.margins(x=0.10)
