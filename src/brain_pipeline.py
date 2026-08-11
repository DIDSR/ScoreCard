"""Entry points for the two brain analyses.

:func:`run_mask_analysis`
    Compares groups of NIfTI segmentation masks. Resolves each group from glob
    patterns, measures the physical volume of every mask, samples a few cases
    per group for a center-slice comparison grid, and summarizes the per-group
    volume distributions. Drives ``notebooks/10_Brain_Mask_Analysis.ipynb``.

:func:`run_cta_analysis`
    Walks an experiment tree of HDF5 CT volumes and their paired masks,
    extracts LBP and GLCM texture features from the masked middle axial slice
    of each case, and projects the feature matrix with PCA. Drives
    ``notebooks/11_Brain_CTA_Analysis.ipynb``.

Both run standalone and both write their figures, tables, and interactive plots
under ``data/notebook_outputs/``. The work itself lives in ``src.brain_masks``,
``src.brain_cta``, and ``src.brain_visualization``; this module only sequences
those calls and names the outputs.
"""

import os

from   src                   import brain_cta, brain_masks, brain_visualization
from   src.notebook_outputs  import (
                                        DEFAULT_OUTPUTS_DIR,
                                        analysis_output_dir,
                                        save_notebook_figures,
                                    )


DEFAULT_MASK_OUTPUT_ID = "brain_masks"
DEFAULT_CTA_OUTPUT_ID  = "brain_cta"


def _banner(text):
    """Print a titled rule so consecutive runs stay legible in one notebook."""
    print("=" * 72)
    print(text)
    print("=" * 72)


def run_mask_analysis(
    input_groups,
    excluded_cases=None,
    inverted_mask_filenames=brain_masks.DEFAULT_INVERTED_MASK_FILENAMES,
    analysis_name="Segmentation Mask",
    num_display=5,
    seed=42,
    zoom_margin=1.18,
    group_colors=None,
    inspect_extremes=True,
    save_outputs=True,
    output_id=DEFAULT_MASK_OUTPUT_ID,
    output_prefix=None,
    base_dir=DEFAULT_OUTPUTS_DIR,
    verbose=True,
) -> dict:
    """Measure, compare, and summarize groups of NIfTI segmentation masks.

    Masks named in ``inverted_mask_filenames`` store background rather than
    foreground, so one call handles conventional masks such as
    ``skull.nii.gz`` and inverted ones such as ``no_brain.nii.gz`` without
    further configuration.

    Args:
        input_groups: Mapping from group name to one path/glob or a list of
            them, for example ``{"Real": ["/path/real/*/skull.nii.gz"]}``.
        excluded_cases: Optional mapping from group name to case identifiers
            (directory names) to skip.
        inverted_mask_filenames: Filenames whose positive voxels mark
            background rather than foreground.
        analysis_name: Label used in figure titles and, unless
            ``output_prefix`` is given, in output filenames.
        num_display: Cases sampled per group for the comparison grid.
        seed: Seed for that sampling.
        zoom_margin: Multiplier applied to the largest mask bounding box when
            choosing the shared display window.
        group_colors: Optional mapping from group name to an explicit color.
        inspect_extremes: When True, add one panel per group showing that
            group's smallest-volume case.
        save_outputs: When True, write figures and tables under
            ``<base_dir>/<output_id>/``.
        output_id: Output sub-directory name.
        output_prefix: Filename prefix for this analysis. Defaults to a
            filesystem-safe form of ``analysis_name``.
        base_dir: Root notebook outputs directory.
        verbose: When True, print progress and the summary tables.

    Returns:
        dict: Keys ``analysis_name``, ``groups`` (resolved paths per group),
        ``selected``, ``records``, ``case_statistics``, ``summary``,
        ``summary_text``, ``figures``, ``group_colors``, ``errors``, and
        ``saved_paths``.

    Raises:
        ValueError: When ``input_groups`` is empty.
        FileNotFoundError: When no group resolves to any NIfTI file.
    """
    if not input_groups:
        raise ValueError(
            "input_groups is required, for example "
            "{'Real': ['/path/real/*/skull.nii.gz']}."
        )

    if verbose:
        _banner(f"{analysis_name.upper()} — VOLUMETRY")

    groups = brain_masks.resolve_input_groups(
        input_groups,
        excluded_cases=excluded_cases,
        verbose=verbose,
    )

    if not groups:
        raise FileNotFoundError(
            "No NIfTI masks were found. Check input_groups: entries must be "
            "exact files or globs such as '/path/**/*.nii.gz'."
        )

    resolved_colors = brain_visualization.resolve_group_colors(groups, group_colors)

    case_statistics, errors = brain_masks.collect_mask_statistics(
        groups,
        inverted_filenames=inverted_mask_filenames,
        verbose=verbose,
    )

    summary      = brain_masks.summarize_mask_statistics(case_statistics)
    summary_text = brain_masks.format_summary_tables(summary)

    selected = brain_masks.select_display_cases(groups, num_display, seed)
    records  = brain_masks.load_display_records(
        selected,
        inverted_filenames=inverted_mask_filenames,
        verbose=verbose,
    )

    figures = {
        "comparison": brain_visualization.plot_mask_grid(
            records,
            group_colors=resolved_colors,
            title=f"{analysis_name} Comparison",
            zoom_margin=zoom_margin,
        ),
        "volume_distribution": brain_visualization.plot_volume_distributions(
            case_statistics,
            group_colors=resolved_colors,
            title=f"{analysis_name} Volume Distribution",
        ),
    }

    if inspect_extremes and not case_statistics.empty:
        smallest    = brain_masks.extreme_case_paths(case_statistics, "min", 1)
        group_means = case_statistics.groupby("group")["volume_cm3"].mean()

        for group_name, paths in smallest.items():
            try:
                record = brain_masks.load_case(
                    paths[0],
                    group_name,
                    inverted_filenames=inverted_mask_filenames,
                )

            except Exception as exc:
                if verbose:
                    print(f"ERROR loading smallest {group_name} case — {exc}")

                continue

            key          = f"smallest_{brain_masks.make_safe_name(group_name)}"
            figures[key] = brain_visualization.plot_extreme_case(
                record,
                group_mean_cm3=float(group_means.get(group_name)),
                title=f"{analysis_name} · {group_name} — Smallest Volume Case",
            )

    if verbose:
        print()
        print(summary_text)

    results = {
        "analysis_name"  : analysis_name,
        "groups"         : groups,
        "selected"       : selected,
        "records"        : records,
        "case_statistics": case_statistics,
        "summary"        : summary,
        "summary_text"   : summary_text,
        "figures"        : {key: fig for key, fig in figures.items() if fig is not None},
        "group_colors"   : resolved_colors,
        "errors"         : errors,
        "saved_paths"    : [],
    }

    if save_outputs:
        results["saved_paths"] = save_mask_outputs(
            results,
            output_id=output_id,
            output_prefix=output_prefix,
            base_dir=base_dir,
            verbose=verbose,
        )

    return results


def save_mask_outputs(
    results,
    output_id=DEFAULT_MASK_OUTPUT_ID,
    output_prefix=None,
    base_dir=DEFAULT_OUTPUTS_DIR,
    verbose=True,
) -> list:
    """Write the figures and tables from a mask analysis.

    Args:
        results: Mapping returned by :func:`run_mask_analysis`.
        output_id: Sub-directory under the notebook outputs directory.
        output_prefix: Filename prefix. Defaults to a filesystem-safe form of
            the analysis name, which keeps several analyses side by side in one
            directory.
        base_dir: Root notebook outputs directory.
        verbose: When True, print the output directory and file count.

    Returns:
        list[str]: Paths of everything written.
    """
    prefix  = output_prefix or brain_masks.make_safe_name(results["analysis_name"])
    out_dir = analysis_output_dir(output_id, base_dir=base_dir)

    saved = save_notebook_figures(
        output_id,
        results["figures"],
        prefix=prefix,
        base_dir=base_dir,
    )

    for name, table in (
        ("case_statistics", results["case_statistics"]),
        ("summary",         results["summary"]),
    ):
        if table is not None and not table.empty:
            path = os.path.join(out_dir, f"{prefix}_{name}.csv")
            table.to_csv(path, index=False)
            saved.append(path)

    summary_path = os.path.join(out_dir, f"{prefix}_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(results["summary_text"] + "\n")

    saved.append(summary_path)

    if verbose:
        print(f"\nSaved {len(saved)} output file(s) to {out_dir}")

    return saved


def run_cta_analysis(
    experiment_root,
    conditions=None,
    max_cases_per_condition=None,
    viewer_condition=None,
    viewer_case=None,
    ct_dir=brain_cta.DEFAULT_CT_DIR,
    ct_dataset=brain_cta.DEFAULT_CT_DATASET,
    mask_dataset=brain_cta.DEFAULT_MASK_DATASET,
    ct_file_template=brain_cta.DEFAULT_CT_FILE_TEMPLATE,
    mask_file_template=brain_cta.DEFAULT_MASK_FILE_TEMPLATE,
    window_center=brain_cta.BRAIN_WINDOW_CENTER,
    window_width=brain_cta.BRAIN_WINDOW_WIDTH,
    lbp_radius=brain_cta.DEFAULT_LBP_RADIUS,
    lbp_n_points=None,
    lbp_method=brain_cta.DEFAULT_LBP_METHOD,
    glcm_distances=brain_cta.DEFAULT_GLCM_DISTANCES,
    glcm_angles=brain_cta.DEFAULT_GLCM_ANGLES,
    glcm_props=brain_cta.DEFAULT_GLCM_PROPS,
    n_components=2,
    save_outputs=True,
    output_id=DEFAULT_CTA_OUTPUT_ID,
    output_prefix="cta",
    base_dir=DEFAULT_OUTPUTS_DIR,
    verbose=True,
) -> dict:
    """Visualize one CTA case, then extract texture features across all cases.

    Args:
        experiment_root: Directory laid out as
            ``<root>/<condition>/<case>/<ct_dir>/``.
        conditions: Optional subset of condition names to process.
        max_cases_per_condition: Optional cap on cases per condition, useful
            for a quick trial run before a full pass.
        viewer_condition: Condition of the single case shown in the three-plane
            viewer. Defaults to the first case discovered.
        viewer_case: Case folder for that viewer, such as ``case_0000``.
        ct_dir: Sub-directory holding the simulated CT files.
        ct_dataset: Dataset key inside the CT file.
        mask_dataset: Dataset key inside the mask file.
        ct_file_template: Template for the CT filename.
        mask_file_template: Template for the mask filename.
        window_center: CT window centre, in HU.
        window_width: CT window width, in HU.
        lbp_radius: LBP radius, in pixels.
        lbp_n_points: LBP sampling points. Defaults to ``8 * lbp_radius``.
        lbp_method: ``local_binary_pattern`` method name.
        glcm_distances: GLCM offsets, in pixels.
        glcm_angles: GLCM angles, in radians.
        glcm_props: GLCM properties to average.
        n_components: Principal components to keep.
        save_outputs: When True, write figures and tables under
            ``<base_dir>/<output_id>/``.
        output_id: Output sub-directory name.
        output_prefix: Filename prefix for the saved outputs.
        base_dir: Root notebook outputs directory.
        verbose: When True, print per-condition progress.

    Returns:
        dict: Keys ``features``, ``skipped``, ``pca``, ``figures``,
        ``pca_figure``, ``viewer``, and ``saved_paths``. ``pca`` and
        ``pca_figure`` are ``None`` when too few cases were extracted to
        project.

    Raises:
        ValueError: When ``experiment_root`` is missing.
        FileNotFoundError: When no case folder exists, or none yields usable
            features.
    """
    if not experiment_root:
        raise ValueError("experiment_root is required for the CTA analysis.")

    if verbose:
        _banner("CTA TEXTURE FEATURES AND PCA")

    figures = {}
    viewer  = None

    pairs = brain_cta.discover_cases(
        experiment_root,
        conditions,
        max_cases_per_condition,
    )

    if not pairs:
        raise FileNotFoundError(f"No case folders found under {experiment_root}.")

    if viewer_condition is None or viewer_case is None:
        viewer_condition, viewer_case = pairs[0]

    ct_path, mask_path = brain_cta.case_files(
        experiment_root,
        viewer_condition,
        viewer_case,
        ct_dir=ct_dir,
        ct_file_template=ct_file_template,
        mask_file_template=mask_file_template,
    )

    try:
        ct_volume, mask_volume = brain_cta.load_case_volumes(
            ct_path,
            mask_path,
            ct_dataset=ct_dataset,
            mask_dataset=mask_dataset,
        )

    except (OSError, KeyError) as exc:
        if verbose:
            print(f"Viewer case skipped ({viewer_condition}/{viewer_case}): {exc}")

    else:
        windowed = brain_cta.apply_window(ct_volume, window_center, window_width)

        viewer = {
            "condition": viewer_condition,
            "case"     : viewer_case,
            "ct_path"  : str(ct_path),
            "mask_path": str(mask_path),
            "ct_shape" : tuple(ct_volume.shape),
        }

        figures["viewer_ct"] = brain_visualization.plot_three_planes(
            windowed,
            title=(
                f"CT Image — {viewer_condition} | {viewer_case} "
                f"(window: C={window_center} W={window_width})"
            ),
            cmap="gray",
            vmin=float(windowed.min()),
            vmax=float(windowed.max()),
        )

        figures["viewer_mask"] = brain_visualization.plot_three_planes(
            mask_volume,
            title=f"Mask: {mask_dataset} — {viewer_condition} | {viewer_case}",
            cmap="Reds",
            vmin=0,
            vmax=1,
        )

    features, skipped = brain_cta.build_feature_table(
        experiment_root,
        conditions=conditions,
        max_cases_per_condition=max_cases_per_condition,
        ct_dir=ct_dir,
        ct_dataset=ct_dataset,
        mask_dataset=mask_dataset,
        ct_file_template=ct_file_template,
        mask_file_template=mask_file_template,
        window_center=window_center,
        window_width=window_width,
        lbp_radius=lbp_radius,
        lbp_n_points=lbp_n_points,
        lbp_method=lbp_method,
        glcm_distances=glcm_distances,
        glcm_angles=glcm_angles,
        glcm_props=glcm_props,
        verbose=verbose,
    )

    if features.empty:
        raise FileNotFoundError(
            "No CTA case produced texture features. Check experiment_root, "
            "ct_dataset, and mask_dataset against the files on disk."
        )

    pca_result = None
    pca_figure = None

    if len(features) >= n_components:
        pca_result = brain_cta.pca_projection(features, n_components=n_components)
        pca_figure = brain_visualization.plot_texture_pca(
            features,
            pca_result,
            subtitle=f"{mask_dataset} mask — all conditions & cases",
        )

    elif verbose:
        print(
            f"PCA skipped: {len(features)} sample(s) is fewer than "
            f"n_components={n_components}."
        )

    results = {
        "features"   : features,
        "skipped"    : skipped,
        "pca"        : pca_result,
        "figures"    : figures,
        "pca_figure" : pca_figure,
        "viewer"     : viewer,
        "saved_paths": [],
    }

    if save_outputs:
        results["saved_paths"] = save_cta_outputs(
            results,
            output_id=output_id,
            output_prefix=output_prefix,
            base_dir=base_dir,
            verbose=verbose,
        )

    return results


def save_cta_outputs(
    results,
    output_id=DEFAULT_CTA_OUTPUT_ID,
    output_prefix="cta",
    base_dir=DEFAULT_OUTPUTS_DIR,
    verbose=True,
) -> list:
    """Write the figures, feature table, and interactive PCA plot.

    Args:
        results: Mapping returned by :func:`run_cta_analysis`.
        output_id: Sub-directory under the notebook outputs directory.
        output_prefix: Filename prefix for the saved outputs.
        base_dir: Root notebook outputs directory.
        verbose: When True, print the output directory and file count.

    Returns:
        list[str]: Paths of everything written.
    """
    out_dir = analysis_output_dir(output_id, base_dir=base_dir)

    saved = save_notebook_figures(
        output_id,
        results["figures"],
        prefix=output_prefix,
        base_dir=base_dir,
    )

    features = results["features"]

    if features is not None and not features.empty:
        path = os.path.join(out_dir, f"{output_prefix}_texture_features.csv")
        features.to_csv(path, index=False)
        saved.append(path)

    if results["pca_figure"] is not None:
        path = os.path.join(out_dir, f"{output_prefix}_texture_pca.html")
        results["pca_figure"].write_html(path)
        saved.append(path)

    if verbose:
        print(f"\nSaved {len(saved)} output file(s) to {out_dir}")

    return saved
