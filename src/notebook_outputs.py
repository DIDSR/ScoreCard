from __future__ import annotations

import glob
import os


DEFAULT_OUTPUTS_DIR = os.path.join("data", "notebook_outputs")

SCORECARD_ANALYSES = (
    "congruence",
    "coverage",
    "completeness",
    "consistency",
    "constraint",
)


def notebook_outputs_dir(base_dir=DEFAULT_OUTPUTS_DIR) -> str:
    """Return the absolute notebook outputs directory, creating it if needed.

    Args:
        base_dir: Root directory for notebook outputs. Defaults to
            ``data/notebook_outputs``.

    Returns:
        str: Absolute path to the notebook outputs directory.
    """
    path = os.path.abspath(base_dir)
    os.makedirs(path, exist_ok=True)

    return path


def analysis_output_dir(analysis_id, base_dir=DEFAULT_OUTPUTS_DIR) -> str:
    """Return the output directory for a scorecard analysis, creating it if needed.

    Args:
        analysis_id: Identifier for the analysis subdirectory, such as
            ``'coverage'`` or ``'congruence'``.
        base_dir: Root notebook outputs directory. Defaults to
            ``data/notebook_outputs``.

    Returns:
        str: Absolute path to the analysis-specific output directory.
    """
    path = os.path.join(notebook_outputs_dir(base_dir), analysis_id)
    os.makedirs(path, exist_ok=True)

    return path


def _safe_metric_name(metric) -> str:
    """Convert a metric name into a filesystem-safe filename token.

    Args:
        metric: Metric name or label to sanitize.

    Returns:
        str: Sanitized metric name with spaces, slashes, and percent signs
        replaced for safe use in filenames.
    """
    return (
        str(metric)
        .replace(" ", "_")
        .replace("/", "-")
        .replace("%", "pct")
    )


def save_notebook_figures(
    analysis_id,
    figs,
    *,
    prefix=None,
    base_dir=DEFAULT_OUTPUTS_DIR,
) -> list[str]:
    """Save one or more matplotlib figures for a notebook analysis.

    Args:
        analysis_id: Identifier for the analysis subdirectory where figures are
            stored.
        figs: A single matplotlib Figure or a dict mapping metric names to
            Figure objects. ``None`` figures in a dict are skipped.
        prefix: Filename prefix for saved PNG files. Defaults to
            ``analysis_id``.
        base_dir: Root notebook outputs directory. Defaults to
            ``data/notebook_outputs``.

    Returns:
        list[str]: Absolute paths to the saved PNG files. Returns an empty list
        when ``figs`` is empty.
    """
    if not figs:
        return []

    out_dir = analysis_output_dir(analysis_id, base_dir=base_dir)
    prefix  = prefix or analysis_id
    saved   = []

    if isinstance(figs, dict):
        for metric, fig in figs.items():
            if fig is None:
                continue

            path = os.path.join(out_dir, f"{prefix}_{_safe_metric_name(metric)}.png")
            fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
            saved.append(path)

    else:
        path = os.path.join(out_dir, f"{prefix}.png")
        figs.savefig(path, bbox_inches="tight", pad_inches=0.12)
        saved.append(path)

    return saved


def list_notebook_figures(
    analysis_id,
    *,
    base_dir=DEFAULT_OUTPUTS_DIR,
) -> list[str]:
    """List saved PNG figures for a notebook analysis.

    Args:
        analysis_id: Identifier for the analysis subdirectory to scan.
        base_dir: Root notebook outputs directory. Defaults to
            ``data/notebook_outputs``.

    Returns:
        list[str]: Sorted absolute paths to PNG files in the analysis output
        directory.
    """
    out_dir = analysis_output_dir(analysis_id, base_dir=base_dir)

    return sorted(glob.glob(os.path.join(out_dir, "*.png")))


def figure_label(path, analysis_id) -> str:
    """Derive a human-readable label from a saved figure filename.

    Args:
        path: Path to a saved figure file.
        analysis_id: Analysis identifier prefix to strip from the filename when
            present.

    Returns:
        str: Display label derived from the basename, with underscores replaced
        by spaces. Falls back to the original basename when stripping leaves an
        empty name.
    """
    name = os.path.splitext(os.path.basename(path))[0]

    for token in (f"{analysis_id}_", analysis_id):
        if name.startswith(token):
            name = name[len(token):]
            break

    return name.replace("_", " ").strip() or os.path.basename(path)