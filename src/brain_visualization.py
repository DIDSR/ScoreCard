"""Figures for the brain pipeline.

Stage 1 (segmentation masks) contributes the center-slice comparison grid, the
volume distribution plot, and the extreme-case inspection panel. Stage 2 (CTA
texture) contributes the three-plane volume viewer and the interactive PCA
scatter.

Every function here returns its figure rather than calling ``plt.show`` or
saving, so notebooks stay in control of display and ``src.brain_pipeline`` stays
in control of file names.
"""

import textwrap

import matplotlib             as     mpl
import matplotlib.pyplot      as     plt
import matplotlib.gridspec    as     gridspec
import numpy                  as     np
import pandas                 as     pd

from   pathlib                import Path
from   scipy                  import stats as scipy_stats

from   src.brain_masks        import VIEW_KEYS, common_display_window, short_case_id


# Qualitative palette used when a group has no explicit color assigned.
DEFAULT_GROUP_PALETTE = (
    "#007CBA",
    "#FF8C42",
    "#222C67",
    "#2E9E6B",
    "#B4508A",
    "#8C6D1F",
)

CARD_FACE        = "#F7F8FA"
TEXT_COLOR       = "#1F2933"
MUTED_COLOR      = "#667085"
IMAGE_BACKGROUND = "#080808"
RULE_COLOR       = "#D9DEE7"
ERROR_COLOR      = "#B42318"

MASK_PLOT_STYLE = {
    "font.family"     : "DejaVu Sans",
    "font.size"       : 10,
    "axes.titleweight": "semibold",
}


def resolve_group_colors(group_names, overrides=None) -> dict:
    """Assign one stable color to each group.

    Args:
        group_names: Iterable of group names, in display order.
        overrides: Optional mapping from group name to an explicit color.

    Returns:
        dict[str, str]: Group name to color, cycling through
        ``DEFAULT_GROUP_PALETTE`` for groups without an override.
    """
    overrides  = dict(overrides or {})
    colors     = {}
    next_index = 0

    for group_name in group_names:
        if group_name in overrides:
            colors[group_name] = overrides[group_name]

        else:
            colors[group_name] = DEFAULT_GROUP_PALETTE[
                next_index % len(DEFAULT_GROUP_PALETTE)
            ]
            next_index += 1

    return colors


def _draw_error_card(fig, inner, record):
    """Fill one grid card with a load-failure message."""
    error_ax = fig.add_subplot(inner[1:6, 1])
    error_ax.axis("off")

    error_ax.text(
        0.5, 0.58, "Could not load mask",
        ha="center", va="center", fontsize=11,
        fontweight="bold", color=ERROR_COLOR,
    )
    error_ax.text(
        0.5, 0.40, textwrap.fill(Path(record["path"]).name, width=28),
        ha="center", va="center", fontsize=9, color=TEXT_COLOR,
    )
    error_ax.text(
        0.5, 0.20, textwrap.fill(record["error"], width=34),
        ha="center", va="center", fontsize=8, color=MUTED_COLOR,
    )


def _draw_case_card(fig, inner, record, group_color, window):
    """Draw the title, three views, and metadata of one loaded case."""
    header_ax = fig.add_subplot(inner[1, 1])
    header_ax.axis("off")

    header_ax.text(
        0.5, 0.56, record["case_id"],
        ha="center", va="center", fontsize=9.8,
        fontweight="bold", color=TEXT_COLOR, linespacing=1.12,
    )
    header_ax.axhline(
        0.04, xmin=0.18, xmax=0.82,
        color=group_color, linewidth=2.0, alpha=0.8,
    )

    for grid_row, view_key in zip((2, 3, 4), VIEW_KEYS):
        view = record["views"][view_key]
        ax   = fig.add_subplot(inner[grid_row, 1])

        ax.set_facecolor(IMAGE_BACKGROUND)
        ax.imshow(
            view["mask"],
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="lower",
            interpolation="nearest",
            extent=view["extent"],
            aspect="equal",
        )

        center_x, center_y           = view["center_mm"]
        window_width, window_height  = window[view_key]

        ax.set_xlim(center_x - window_width / 2.0, center_x + window_width / 2.0)
        ax.set_ylim(center_y - window_height / 2.0, center_y + window_height / 2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_anchor("C")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(
            0.025, 0.96, view["label"],
            transform=ax.transAxes,
            ha="left", va="top", fontsize=8.8,
            fontweight="semibold", color="white",
            bbox={
                "boxstyle" : "round,pad=0.24",
                "facecolor": (0, 0, 0, 0.60),
                "edgecolor": "none",
            },
        )

    info_ax = fig.add_subplot(inner[5, 1])
    info_ax.axis("off")
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.axhline(0.98, xmin=0.03, xmax=0.97, color=RULE_COLOR, linewidth=0.8)

    stats = [
        ("Shape",       " × ".join(str(value) for value in record["shape"])),
        ("Voxel size",  " × ".join(f"{value:.2f}" for value in record["voxel_dims"]) + " mm"),
        ("Mask voxels", f"{record['voxel_count']:,}"),
        ("Volume",      f"{record['volume_cm3']:,.2f} cm³"),
    ]

    for y, (label, value) in zip((0.81, 0.60, 0.39, 0.18), stats):
        info_ax.text(
            0.04, y, label,
            ha="left", va="center", fontsize=8.6,
            fontweight="semibold", color=MUTED_COLOR,
        )
        info_ax.text(
            0.96, y, value,
            ha="right", va="center", fontsize=8.6,
            fontfamily="DejaVu Sans Mono", color=TEXT_COLOR,
        )


def plot_mask_grid(
    records_by_group,
    group_colors=None,
    title="Segmentation Mask Comparison",
    subtitle="Center slices — common physical zoom within each view",
    zoom_margin=1.18,
):
    """Draw one card per sampled case, arranged as one row per group.

    Every panel of a given view shares a physical zoom window, so mask sizes
    can be compared directly across cards.

    Args:
        records_by_group: Mapping from group name to its case records, as
            returned by ``src.brain_masks.load_display_records``.
        group_colors: Optional mapping from group name to color.
        title: Figure title.
        subtitle: Line printed under the title.
        zoom_margin: Multiplier applied to the largest bounding box.

    Returns:
        matplotlib.figure.Figure: The comparison grid.

    Raises:
        RuntimeError: When no case in ``records_by_group`` loaded successfully.
    """
    selected_items = [
        (group_name, records)
        for group_name, records in records_by_group.items()
        if records
    ]

    if not any(
        "error" not in record
        for _, records in selected_items
        for record in records
    ):
        raise RuntimeError("No selected masks could be loaded for plotting.")

    group_colors = group_colors or resolve_group_colors(
        [group_name for group_name, _ in selected_items]
    )

    window = common_display_window(records_by_group, zoom_margin)

    nrows = len(selected_items)
    ncols = max(len(records) for _, records in selected_items)

    with mpl.rc_context(MASK_PLOT_STYLE):
        fig = plt.figure(
            figsize=(max(11.0, 3.25 * ncols + 0.9), 5.35 * nrows + 1.0),
            facecolor="white",
        )

        fig.suptitle(
            title,
            fontsize=20,
            fontweight="bold",
            color=TEXT_COLOR,
            y=0.975,
        )
        fig.text(
            0.5, 0.943, subtitle,
            ha="center", va="center",
            fontsize=10.5, color=MUTED_COLOR,
        )

        outer = fig.add_gridspec(
            nrows=nrows,
            ncols=ncols + 1,
            width_ratios=[0.16] + [1] * ncols,
            left=0.025,
            right=0.99,
            bottom=0.035,
            top=0.905,
            wspace=0.10,
            hspace=0.22,
        )

        for row_idx, (group_name, records) in enumerate(selected_items):
            group_color  = group_colors.get(group_name, DEFAULT_GROUP_PALETTE[0])
            row_label_ax = fig.add_subplot(outer[row_idx, 0])
            row_label_ax.axis("off")

            row_label_ax.text(
                0.5, 0.5, str(group_name).upper(),
                rotation=90, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color="white",
                bbox={
                    "boxstyle" : "round,pad=0.55",
                    "facecolor": group_color,
                    "edgecolor": "none",
                },
            )

            for col_idx in range(ncols):
                if col_idx >= len(records):
                    blank_ax = fig.add_subplot(outer[row_idx, col_idx + 1])
                    blank_ax.axis("off")

                    continue

                record  = records[col_idx]
                card_ax = fig.add_subplot(outer[row_idx, col_idx + 1])

                card_ax.set_facecolor(CARD_FACE)
                card_ax.set_xticks([])
                card_ax.set_yticks([])
                card_ax.set_zorder(-1)

                for spine in card_ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(group_color)
                    spine.set_linewidth(1.0)
                    spine.set_alpha(0.55)

                inner = gridspec.GridSpecFromSubplotSpec(
                    7,
                    3,
                    subplot_spec=outer[row_idx, col_idx + 1],
                    # Top margin, title, three views, metadata, bottom margin.
                    height_ratios=[0.10, 0.78, 2.10, 2.10, 2.10, 1.58, 0.10],
                    width_ratios=[0.045, 1.0, 0.045],
                    hspace=0.10,
                    wspace=0.0,
                )

                if "error" in record:
                    _draw_error_card(fig, inner, record)

                else:
                    _draw_case_card(fig, inner, record, group_color, window)

    return fig


def confidence_interval_95(values):
    """Return the 95% confidence interval of a sample mean.

    Args:
        values: Sample values; missing values are dropped.

    Returns:
        tuple[float, float] | None: The interval, or ``None`` when fewer than
        two usable values remain or the standard error is not finite.
    """
    series = pd.Series(values).dropna()

    if len(series) < 2:
        return None

    standard_error = scipy_stats.sem(series)

    if not np.isfinite(standard_error):
        return None

    return scipy_stats.t.interval(
        0.95,
        df=len(series) - 1,
        loc=series.mean(),
        scale=standard_error,
    )


def plot_volume_distributions(
    df,
    group_colors=None,
    title="Segmentation Mask Volume Distribution",
    bins=30,
):
    """Overlay per-group volume histograms with KDE curves, means, and 95% CIs.

    Args:
        df: Per-case DataFrame with ``group`` and ``volume_cm3`` columns.
        group_colors: Optional mapping from group name to color.
        title: Figure title.
        bins: Number of histogram bins, shared by every group.

    Returns:
        matplotlib.figure.Figure | None: The distribution figure, or ``None``
        when ``df`` holds no volumes.
    """
    if df is None or df.empty:
        return None

    volumes_all = df["volume_cm3"].dropna()

    if volumes_all.empty:
        return None

    group_names  = list(dict.fromkeys(df["group"]))
    group_colors = group_colors or resolve_group_colors(group_names)

    global_min = float(volumes_all.min())
    global_max = float(volumes_all.max())

    if np.isclose(global_min, global_max):
        padding    = max(abs(global_min) * 0.05, 1.0)
        global_min = global_min - padding
        global_max = global_max + padding

    bin_edges = np.linspace(global_min, global_max, bins)
    x_range   = np.linspace(global_min, global_max, 300)

    with mpl.rc_context(MASK_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))

        fig.patch.set_facecolor("white")
        ax.set_facecolor("#F8F8F8")
        ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)

        for group_name in group_names:
            volumes = df.loc[df["group"] == group_name, "volume_cm3"].dropna()

            if volumes.empty:
                continue

            color = group_colors.get(group_name, DEFAULT_GROUP_PALETTE[0])

            ax.hist(
                volumes,
                bins=bin_edges,
                density=True,
                alpha=0.35,
                color=color,
                label=f"{group_name} (n={len(volumes)})",
                zorder=2,
            )

            if len(volumes) > 1 and volumes.nunique() > 1:
                try:
                    kde = scipy_stats.gaussian_kde(volumes)
                    ax.plot(x_range, kde(x_range), color=color, linewidth=2, zorder=3)

                except np.linalg.LinAlgError:
                    print(f"KDE skipped for {group_name}: singular covariance.")

            ax.axvline(
                volumes.mean(),
                color=color,
                linestyle="--",
                linewidth=1.5,
                zorder=4,
            )

            interval = confidence_interval_95(volumes)

            if interval is not None and np.all(np.isfinite(interval)):
                ax.axvspan(*interval, alpha=0.10, color=color, zorder=1)

        ax.set_xlabel("Volume (cm³)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)

        fig.tight_layout()

    return fig


def plot_extreme_case(record, group_mean_cm3=None, title=None, cmap="hot"):
    """Show the three center slices of a single case for closer inspection.

    Typically used on the smallest-volume case of a group to check whether a
    low volume comes from a genuinely small mask or a failed segmentation.

    Args:
        record: Case record from ``src.brain_masks.load_case``.
        group_mean_cm3: Optional group mean volume, printed for comparison.
        title: Figure title. Defaults to a title naming the group.
        cmap: Colormap for the mask slices.

    Returns:
        matplotlib.figure.Figure: The inspection panel.
    """
    title = title or f"{record.get('group', 'Case')} — Volume Inspection"

    with mpl.rc_context(MASK_PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor("white")

        for ax, view_key in zip(axes, VIEW_KEYS):
            view = record["views"][view_key]

            ax.imshow(view["mask"], cmap=cmap, origin="lower", aspect="equal")
            ax.set_title(view["label"], fontsize=10, color="#444444")
            ax.axis("off")

        info_lines = [
            f"Case ID:      {short_case_id(record['path'])}",
            f"Voxel size:   "
            f"{tuple(round(value, 4) for value in record['voxel_dims'])} mm",
            f"Voxel count:  {record['voxel_count']:,}",
            f"Volume:       {record['volume_cm3']:.2f} cm³",
        ]

        if group_mean_cm3 is not None:
            info_lines[-1] += f"  (group mean: {group_mean_cm3:.2f} cm³)"

        fig.suptitle(title, fontsize=13, fontweight="bold", color=TEXT_COLOR)
        fig.tight_layout()

        # Anchored from its top edge so the block hangs below the axes; a
        # 'tight' bounding box on save then grows to include it.
        fig.text(
            0.5, -0.02, "\n".join(info_lines),
            ha="center", va="top", fontsize=9, fontfamily="monospace",
            bbox={
                "boxstyle" : "round",
                "facecolor": "#F4F4F4",
                "edgecolor": "#007CBA",
                "linewidth": 1.5,
                "alpha"    : 0.9,
            },
        )

    return fig


def plot_three_planes(volume, title, cmap="gray", vmin=None, vmax=None):
    """Plot the axial, coronal, and sagittal mid-slices of a ``(Z, Y, X)`` volume.

    Args:
        volume: 3-D array ordered ``(Z, Y, X)``.
        title: Figure title.
        cmap: Colormap.
        vmin: Lower display limit.
        vmax: Upper display limit.

    Returns:
        matplotlib.figure.Figure: The three-plane view.
    """
    volume     = np.asarray(volume)
    z, y, x    = volume.shape

    slices = {
        f"Axial\n(Z = {z // 2})"   : volume[z // 2, :, :],
        f"Coronal\n(Y = {y // 2})" : volume[:, y // 2, :],
        f"Sagittal\n(X = {x // 2})": volume[:, :, x // 2],
    }

    with mpl.rc_context(MASK_PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        for ax, (plane_label, plane) in zip(axes, slices.items()):
            ax.imshow(plane, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(plane_label)
            ax.axis("off")

        fig.tight_layout()

    return fig


def plot_texture_pca(
    features_df,
    pca_result,
    title="PCA of LBP + GLCM Texture Features",
    subtitle="all conditions & cases",
    width=950,
    height=650,
):
    """Build the interactive PCA scatter, one trace per condition.

    Args:
        features_df: Feature table with ``condition`` and ``case`` columns.
        pca_result: Result of ``src.brain_cta.pca_projection``.
        title: Plot title.
        subtitle: Smaller line under the title.
        width: Figure width, in pixels.
        height: Figure height, in pixels.

    Returns:
        plotly.graph_objects.Figure: Scatter of PC1 against PC2, with each
        condition toggleable from the legend.
    """
    # Imported here so stage 1 (masks) does not require plotly.
    import plotly.graph_objects as go

    components = pca_result["components"]
    variance   = pca_result["explained_variance_ratio"] * 100.0

    conditions = features_df["condition"].to_numpy()
    cases      = features_df["case"].to_numpy()

    fig = go.Figure()

    for condition in sorted(set(conditions)):
        selector = conditions == condition

        fig.add_trace(go.Scatter(
            x=components[selector, 0],
            y=components[selector, 1],
            mode="markers+text",
            name=str(condition),
            text=cases[selector],
            textposition="top right",
            textfont={"size": 9},
            marker={"size": 10, "line": {"width": 0.5, "color": "white"}},
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"Condition: {condition}<br>"
                "PC1: %{x:.3f}<br>"
                "PC2: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "font": {"size": 15}},
        xaxis_title=f"PC1 ({variance[0]:.1f}% variance)",
        yaxis_title=f"PC2 ({variance[1]:.1f}% variance)",
        legend={
            "title"          : "Condition",
            "itemclick"      : "toggle",
            "itemdoubleclick": "toggleothers",
        },
        width=width,
        height=height,
        template="plotly_white",
    )

    fig.add_hline(y=0, line={"color": "gray", "width": 0.8, "dash": "dash"})
    fig.add_vline(x=0, line={"color": "gray", "width": 0.8, "dash": "dash"})

    return fig
