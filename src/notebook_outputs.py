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
    path = os.path.abspath(base_dir)
    os.makedirs(path, exist_ok=True)

    return path


def analysis_output_dir(analysis_id, base_dir=DEFAULT_OUTPUTS_DIR) -> str:
    path = os.path.join(notebook_outputs_dir(base_dir), analysis_id)
    os.makedirs(path, exist_ok=True)

    return path


def _safe_metric_name(metric) -> str:
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
    out_dir = analysis_output_dir(analysis_id, base_dir=base_dir)

    return sorted(glob.glob(os.path.join(out_dir, "*.png")))


def figure_label(path, analysis_id) -> str:
    name = os.path.splitext(os.path.basename(path))[0]

    for token in (f"{analysis_id}_", analysis_id):
        if name.startswith(token):
            name = name[len(token):]
            break

    return name.replace("_", " ").strip() or os.path.basename(path)