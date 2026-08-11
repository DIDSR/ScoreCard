"""NIfTI segmentation-mask discovery, measurement, and summary statistics.

This is the data layer for stage 1 of the brain pipeline (see
``src.brain_pipeline``). It resolves glob patterns into groups of mask files,
converts each mask into a Boolean foreground volume, measures its physical
volume, and aggregates per-group distribution statistics.

Nothing here draws figures; see ``src.brain_visualization``.
"""

import glob
import os
import random
import re
import textwrap

import numpy    as     np
import pandas   as     pd
import nibabel  as     nib

from   pathlib  import Path


# Masks whose filename appears here store the *background* as their positive
# values, so the foreground test is inverted when they are loaded.
DEFAULT_INVERTED_MASK_FILENAMES = ("no_brain.nii.gz",)

NIFTI_SUFFIXES = (".nii", ".nii.gz")

VIEW_KEYS = ("axial", "coronal", "sagittal")

_GLOB_MAGIC_PATTERN = re.compile(r"[*?\[]")


def strip_nifti_suffix(filename) -> str:
    """Remove a ``.nii`` or ``.nii.gz`` suffix from a filename.

    Args:
        filename: File name, with or without a NIfTI suffix.

    Returns:
        str: The filename without its NIfTI suffix.
    """
    lower = str(filename).casefold()

    if lower.endswith(".nii.gz"):
        return str(filename)[:-7]

    if lower.endswith(".nii"):
        return str(filename)[:-4]

    return str(filename)


def make_safe_name(value, fallback="brain_pipeline") -> str:
    """Convert a label into a filesystem-safe output name.

    Args:
        value: Label to sanitize, such as an analysis or group name.
        fallback: Name returned when sanitizing leaves an empty string.

    Returns:
        str: Sanitized name containing only ``A-Z a-z 0-9 . _ -``.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())

    return safe.strip("._") or fallback


def _has_glob_magic(pattern) -> bool:
    """Return True when a path string contains glob wildcards."""
    return bool(_GLOB_MAGIC_PATTERN.search(str(pattern)))


def expand_path_entry(entry, verbose=True) -> list:
    """Resolve one exact path or glob pattern into existing NIfTI files.

    Directories are not scanned recursively on their own; pass an explicit
    glob such as ``/path/**/*.nii.gz`` instead.

    Args:
        entry: Exact file path or glob pattern. ``~`` and environment
            variables are expanded.
        verbose: When True, print a warning for entries that resolve to
            nothing usable.

    Returns:
        list[str]: Sorted paths of existing files with a NIfTI suffix.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(entry)))

    if _has_glob_magic(expanded):
        matches = sorted(glob.glob(expanded, recursive=True))

    else:
        path = Path(expanded)

        if path.is_file():
            matches = [str(path)]

        elif path.is_dir():
            if verbose:
                print(
                    f"WARNING: directory paths are not scanned automatically: {path}\n"
                    "         Provide exact NIfTI files or a glob such as "
                    "'/path/**/*.nii.gz'."
                )

            matches = []

        else:
            if verbose:
                print(f"WARNING: path does not exist: {path}")

            matches = []

    return [
        str(Path(match))
        for match in matches
        if Path(match).is_file()
        and Path(match).name.casefold().endswith(NIFTI_SUFFIXES)
    ]


def resolve_input_groups(input_groups, excluded_cases=None, verbose=True) -> dict:
    """Expand path patterns per group, drop duplicates, and apply exclusions.

    A case is excluded when any directory name along its path matches an entry
    in ``excluded_cases`` for that group.

    Args:
        input_groups: Mapping from group name to one path/pattern or a list of
            them, for example
            ``{"Real": ["/path/real/*/skull.nii.gz"], "Synthetic": [...]}``.
        excluded_cases: Optional mapping from group name to a collection of
            case identifiers (directory names) to skip.
        verbose: When True, print how many masks were found and excluded.

    Returns:
        dict[str, list[str]]: Group name to its sorted, deduplicated,
        non-excluded mask paths. Groups that resolve to no files are dropped.
    """
    excluded_cases = excluded_cases or {}
    resolved       = {}

    for group_name, entries in input_groups.items():
        group_name = str(group_name)

        if isinstance(entries, (str, Path)):
            entries = [entries]

        group_exclusions  = set(excluded_cases.get(group_name, set()))

        paths             = []
        seen              = set()
        excluded_count    = 0
        excluded_case_ids = set()

        for entry in entries:
            for mask_path in expand_path_entry(entry, verbose=verbose):
                mask_path = Path(mask_path)

                matching_exclusions = group_exclusions.intersection(mask_path.parts)

                if matching_exclusions:
                    excluded_count += 1
                    excluded_case_ids.update(matching_exclusions)

                    continue

                normalized = str(mask_path.resolve())

                if normalized not in seen:
                    seen.add(normalized)
                    paths.append(normalized)

        if not paths:
            if verbose:
                print(f"Found    0 {group_name.lower()} mask(s) — group skipped")

            continue

        resolved[group_name] = sorted(paths)

        if verbose:
            print(f"Found {len(resolved[group_name]):>4} {group_name.lower()} mask(s)")

            if excluded_count:
                print(
                    f"Excluded {excluded_count} {group_name.lower()} mask(s): "
                    f"{', '.join(sorted(excluded_case_ids))}"
                )

    return resolved


def requires_mask_inversion(
    mask_path,
    inverted_filenames=DEFAULT_INVERTED_MASK_FILENAMES,
) -> bool:
    """Return True when a mask stores background rather than foreground.

    Args:
        mask_path: Path to a NIfTI mask.
        inverted_filenames: Filenames whose positive voxels mark background,
            such as ``no_brain.nii.gz``.

    Returns:
        bool: True when the foreground test must be inverted for this file.
    """
    names = {str(name).casefold() for name in inverted_filenames}

    return Path(mask_path).name.casefold() in names


def prepare_foreground_mask(
    img,
    mask_path,
    inverted_filenames=DEFAULT_INVERTED_MASK_FILENAMES,
) -> np.ndarray:
    """Convert a loaded NIfTI image into a Boolean foreground mask.

    For inverted filenames, ``value <= 0`` is foreground; otherwise
    ``value > 0`` is foreground. Non-finite voxels are always background.

    Args:
        img: Image returned by ``nibabel.load``.
        mask_path: Path the image was loaded from, used to detect inversion.
        inverted_filenames: Filenames that require inversion.

    Returns:
        numpy.ndarray: Boolean array with the same shape as the image.
    """
    raw_data = np.asanyarray(img.dataobj)
    finite   = np.isfinite(raw_data)

    if requires_mask_inversion(mask_path, inverted_filenames):
        foreground = finite & (raw_data <= 0)

    else:
        foreground = finite & (raw_data > 0)

    return foreground.astype(bool, copy=False)


def measure_mask(
    img,
    mask_path,
    inverted_filenames=DEFAULT_INVERTED_MASK_FILENAMES,
) -> dict:
    """Measure the foreground volume of one loaded mask.

    Args:
        img: Image returned by ``nibabel.load``.
        mask_path: Path the image was loaded from.
        inverted_filenames: Filenames that require inversion.

    Returns:
        dict: Keys ``mask`` (Boolean 3-D array), ``voxel_dims`` (mm),
        ``voxel_volume_mm3``, ``voxel_count``, ``volume_mm3``, and
        ``volume_cm3``.
    """
    mask             = prepare_foreground_mask(img, mask_path, inverted_filenames)
    voxel_dims       = tuple(float(value) for value in img.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(voxel_dims))
    voxel_count      = int(np.count_nonzero(mask))
    volume_mm3       = voxel_count * voxel_volume_mm3

    return {
        "mask"            : mask,
        "voxel_dims"      : voxel_dims,
        "voxel_volume_mm3": voxel_volume_mm3,
        "voxel_count"     : voxel_count,
        "volume_mm3"      : volume_mm3,
        "volume_cm3"      : volume_mm3 / 1000.0,
    }


def short_case_id(mask_path, max_length=50) -> str:
    """Build a compact, layout-independent case identifier.

    Args:
        mask_path: Path to a mask file.
        max_length: Maximum length before the identifier is truncated.

    Returns:
        str: ``grandparent / parent / filename`` where those parts exist,
        truncated with an ellipsis when longer than ``max_length``.
    """
    path        = Path(mask_path)
    filename    = strip_nifti_suffix(path.name)
    parent      = path.parent.name
    grandparent = path.parent.parent.name if path.parent.parent != path.parent else ""

    if grandparent and parent:
        case_id = f"{grandparent} / {parent} / {filename}"

    elif parent:
        case_id = f"{parent} / {filename}"

    else:
        case_id = filename

    if len(case_id) > max_length:
        case_id = case_id[: max_length - 1] + "…"

    return case_id


def wrap_case_id(mask_path, width=30, max_length=50) -> str:
    """Wrap a case identifier onto at most two lines for figure titles.

    Args:
        mask_path: Path to a mask file.
        width: Maximum characters per line.
        max_length: Maximum identifier length before truncation.

    Returns:
        str: Wrapped identifier of no more than two lines.
    """
    return textwrap.fill(
        short_case_id(mask_path, max_length=max_length),
        width=width,
        max_lines=2,
        placeholder="…",
        break_long_words=True,
        break_on_hyphens=False,
    )


def get_center_slices(mask):
    """Return the three orthogonal slices through the foreground center.

    Args:
        mask: Boolean 3-D array.

    Returns:
        tuple: ``(axial, coronal, sagittal, (cx, cy, cz))`` where the slices are
        2-D Boolean arrays and the centre falls back to the volume midpoint for
        an empty mask.
    """
    mask                   = np.asarray(mask, dtype=bool)
    foreground_coordinates = np.argwhere(mask)

    if foreground_coordinates.size == 0:
        cx, cy, cz = [dimension // 2 for dimension in mask.shape]

    else:
        cx, cy, cz = foreground_coordinates.mean(axis=0).astype(int)

    return (
        mask[:, :, cz],
        mask[:, cy, :],
        mask[cx, :, :],
        (cx, cy, cz),
    )


def foreground_geometry(view, x_spacing, y_spacing):
    """Calculate physical geometry for one 2-D Boolean slice.

    Args:
        view: 2-D array-like slice.
        x_spacing: Voxel spacing along the displayed x axis, in millimetres.
        y_spacing: Voxel spacing along the displayed y axis, in millimetres.

    Returns:
        tuple: ``(mask, extent, center_mm, bbox_mm)``. ``extent`` is an
        ``imshow`` extent in millimetres, ``center_mm`` is the foreground
        bounding-box centre, and ``bbox_mm`` is its width and height, or
        ``None`` when the slice has no foreground.
    """
    mask          = np.asarray(view, dtype=bool)
    height, width = mask.shape
    extent        = (0.0, width * x_spacing, 0.0, height * y_spacing)
    yy, xx        = np.where(mask)

    if xx.size == 0:
        center_mm = (extent[1] / 2.0, extent[3] / 2.0)
        bbox_mm   = None

    else:
        x0 = xx.min() * x_spacing
        x1 = (xx.max() + 1) * x_spacing
        y0 = yy.min() * y_spacing
        y1 = (yy.max() + 1) * y_spacing

        center_mm = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        bbox_mm   = (
            max(x1 - x0, x_spacing),
            max(y1 - y0, y_spacing),
        )

    return mask, extent, center_mm, bbox_mm


def load_case(
    mask_path,
    group_name,
    inverted_filenames=DEFAULT_INVERTED_MASK_FILENAMES,
    case_id_width=30,
    case_id_max_length=50,
) -> dict:
    """Load one mask and prepare its metadata and three display views.

    Args:
        mask_path: Path to a NIfTI mask.
        group_name: Group the case belongs to, such as ``"Real"``.
        inverted_filenames: Filenames that require inversion.
        case_id_width: Wrap width for the display title.
        case_id_max_length: Maximum case identifier length.

    Returns:
        dict: Case record with ``path``, ``group``, ``case_id``, ``shape``,
        ``voxel_dims``, ``voxel_count``, ``volume_cm3``, ``inverted``,
        ``center_voxel``, and a ``views`` mapping keyed by ``VIEW_KEYS``. Each
        view holds its ``mask``, ``extent``, ``center_mm``, ``bbox_mm``, and
        ``label``.
    """
    img         = nib.load(mask_path)
    measurement = measure_mask(img, mask_path, inverted_filenames)
    mask_3d     = measurement["mask"]

    axial, coronal, sagittal, center = get_center_slices(mask_3d)

    dx, dy, dz = measurement["voxel_dims"]

    raw_views = [
        ("axial",    axial.T,    f"Axial  ·  z = {center[2]}",    dx, dy),
        ("coronal",  coronal.T,  f"Coronal  ·  y = {center[1]}",  dx, dz),
        ("sagittal", sagittal.T, f"Sagittal  ·  x = {center[0]}", dy, dz),
    ]

    views = {}

    for key, view, label, x_spacing, y_spacing in raw_views:
        view_mask, extent, center_mm, bbox_mm = foreground_geometry(
            view,
            x_spacing,
            y_spacing,
        )

        views[key] = {
            "mask"     : view_mask,
            "extent"   : extent,
            "center_mm": center_mm,
            "bbox_mm"  : bbox_mm,
            "label"    : label,
        }

    return {
        "path"        : str(mask_path),
        "group"       : group_name,
        "case_id"     : wrap_case_id(mask_path, case_id_width, case_id_max_length),
        "shape"       : tuple(mask_3d.shape),
        "voxel_dims"  : measurement["voxel_dims"],
        "voxel_count" : measurement["voxel_count"],
        "volume_cm3"  : measurement["volume_cm3"],
        "inverted"    : requires_mask_inversion(mask_path, inverted_filenames),
        "center_voxel": center,
        "views"       : views,
    }


def select_display_cases(groups, num_display=5, seed=42) -> dict:
    """Draw a reproducible sample of cases from each group for display.

    Args:
        groups: Mapping from group name to its list of mask paths.
        num_display: Maximum cases to sample per group.
        seed: Seed for the sampling generator.

    Returns:
        dict[str, list[str]]: Group name to its sorted sample of mask paths.
    """
    generator = random.Random(seed)
    selected  = {}

    for group_name, paths in groups.items():
        sample                 = generator.sample(paths, min(num_display, len(paths)))
        selected[group_name]   = sorted(sample)

    return selected


def load_display_records(
    selected_groups,
    inverted_filenames=DEFAULT_INVERTED_MASK_FILENAMES,
    verbose=True,
) -> dict:
    """Load the sampled cases for each group, recording per-case failures.

    Args:
        selected_groups: Mapping from group name to the mask paths to load.
        inverted_filenames: Filenames that require inversion.
        verbose: When True, print load failures as they happen.

    Returns:
        dict[str, list[dict]]: Group name to its case records. A record that
        failed to load holds ``path``, ``group``, and ``error`` only.
    """
    records = {}

    for group_name, paths in selected_groups.items():
        group_records = []

        for mask_path in paths:
            try:
                group_records.append(
                    load_case(mask_path, group_name, inverted_filenames)
                )

            except Exception as exc:
                if verbose:
                    print(f"ERROR loading {mask_path} — {exc}")

                group_records.append({
                    "path" : str(mask_path),
                    "group": group_name,
                    "error": str(exc),
                })

        records[group_name] = group_records

    return records


def common_display_window(records_by_group, zoom_margin=1.18) -> dict:
    """Find one physical zoom window per view, shared by every displayed case.

    Args:
        records_by_group: Mapping from group name to its case records.
        zoom_margin: Multiplier applied to the largest bounding box so masks do
            not touch the panel edges.

    Returns:
        dict[str, tuple[float, float]]: View key to ``(width_mm, height_mm)``.
    """
    valid_records = [
        record
        for group_records in records_by_group.values()
        for record in group_records
        if "error" not in record
    ]

    window = {}

    for view_key in VIEW_KEYS:
        bbox_sizes = [
            record["views"][view_key]["bbox_mm"]
            for record in valid_records
            if record["views"][view_key]["bbox_mm"] is not None
        ]

        if bbox_sizes:
            window[view_key] = (
                max(size[0] for size in bbox_sizes) * zoom_margin,
                max(size[1] for size in bbox_sizes) * zoom_margin,
            )

        else:
            window[view_key] = (1.0, 1.0)

    return window


def collect_mask_statistics(
    groups,
    inverted_filenames=DEFAULT_INVERTED_MASK_FILENAMES,
    verbose=True,
):
    """Measure every mask in every group.

    Args:
        groups: Mapping from group name to its list of mask paths.
        inverted_filenames: Filenames that require inversion.
        verbose: When True, print read failures and a completion count.

    Returns:
        tuple: ``(df, errors)`` where ``df`` is a per-case DataFrame with
        columns ``group``, ``path``, ``filename``, ``case_id``, ``inverted``,
        ``voxel_volume_mm3``, ``voxel_count``, and ``volume_cm3``, and
        ``errors`` is a list of ``(path, message)`` pairs.
    """
    records = []
    errors  = []

    for group_name, paths in groups.items():
        for mask_path in paths:
            try:
                img         = nib.load(mask_path)
                measurement = measure_mask(img, mask_path, inverted_filenames)

                records.append({
                    "group"           : group_name,
                    "path"            : mask_path,
                    "filename"        : Path(mask_path).name,
                    "case_id"         : short_case_id(mask_path),
                    "inverted"        : requires_mask_inversion(mask_path, inverted_filenames),
                    "voxel_volume_mm3": measurement["voxel_volume_mm3"],
                    "voxel_count"     : measurement["voxel_count"],
                    "volume_cm3"      : measurement["volume_cm3"],
                })

            except Exception as exc:
                errors.append((str(mask_path), str(exc)))

                if verbose:
                    print(f"ERROR: {mask_path} — {exc}")

    if verbose:
        print(f"Measured {len(records)} mask(s); {len(errors)} failed.")

    return pd.DataFrame(records), errors


def summarize_mask_statistics(df) -> pd.DataFrame:
    """Aggregate per-case volumes into per-group distribution statistics.

    Args:
        df: Per-case DataFrame from :func:`collect_mask_statistics`.

    Returns:
        pandas.DataFrame: One row per group with sample size, voxel averages,
        and volume mean, median, standard deviation, variance, min, Q1, Q3,
        max, IQR, range, coefficient of variation, and skewness. Statistics
        that need more cases than a group has are ``NaN``.
    """
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("group", sort=False)
        .agg(
            n_cases              =("group",            "size"),
            n_inverted           =("inverted",         "sum"),
            avg_voxel_volume_mm3 =("voxel_volume_mm3", "mean"),
            avg_voxel_count      =("voxel_count",      "mean"),
            mean_volume_cm3      =("volume_cm3",       "mean"),
            median_volume_cm3    =("volume_cm3",       "median"),
            std_volume_cm3       =("volume_cm3",       "std"),
            variance_volume_cm6  =("volume_cm3",       "var"),
            min_volume_cm3       =("volume_cm3",       "min"),
            q1_volume_cm3        =("volume_cm3",       lambda values: values.quantile(0.25)),
            q3_volume_cm3        =("volume_cm3",       lambda values: values.quantile(0.75)),
            max_volume_cm3       =("volume_cm3",       "max"),
            skewness             =("volume_cm3",       "skew"),
        )
        .reset_index()
    )

    summary["iqr_volume_cm3"]   = summary["q3_volume_cm3"] - summary["q1_volume_cm3"]
    summary["range_volume_cm3"] = summary["max_volume_cm3"] - summary["min_volume_cm3"]
    summary["cv_percent"]       = np.where(
        summary["mean_volume_cm3"].ne(0),
        100.0 * summary["std_volume_cm3"] / summary["mean_volume_cm3"],
        np.nan,
    )

    # Keep the derived statistics next to the volume statistics they extend.
    return summary[
        [
            "group",
            "n_cases",
            "n_inverted",
            "avg_voxel_volume_mm3",
            "avg_voxel_count",
            "mean_volume_cm3",
            "median_volume_cm3",
            "std_volume_cm3",
            "variance_volume_cm6",
            "min_volume_cm3",
            "q1_volume_cm3",
            "q3_volume_cm3",
            "max_volume_cm3",
            "iqr_volume_cm3",
            "range_volume_cm3",
            "cv_percent",
            "skewness",
        ]
    ]


def _format_number(value, decimals=2) -> str:
    """Format one numeric cell for the plain-text summary tables."""
    if pd.isna(value):
        return "NA"

    return f"{value:,.{decimals}f}"


def format_summary_tables(summary) -> str:
    """Render the per-group summary as two fixed-width text tables.

    Args:
        summary: Summary DataFrame from :func:`summarize_mask_statistics`.

    Returns:
        str: A sample/voxel table followed by a volume-distribution table,
        ready to ``print``.
    """
    if summary is None or summary.empty:
        return "No valid masks were available for summary statistics."

    group_width = max(14, min(30, int(summary["group"].astype(str).str.len().max())))
    lines       = []

    # Table 1: sample and voxel information.
    voxel_widths = [group_width, 10, 12, 24, 22]
    voxel_sep    = "-" * sum(voxel_widths)

    lines.append("SAMPLE AND VOXEL SUMMARY")
    lines.append(voxel_sep)
    lines.append(
        f"{'Group':<{voxel_widths[0]}}"
        f"{'N Cases':>{voxel_widths[1]}}"
        f"{'Inverted':>{voxel_widths[2]}}"
        f"{'Avg Voxel Vol (mm³)':>{voxel_widths[3]}}"
        f"{'Avg Voxel Count':>{voxel_widths[4]}}"
    )
    lines.append(voxel_sep)

    for _, row in summary.iterrows():
        lines.append(
            f"{str(row['group']):<{voxel_widths[0]}}"
            f"{int(row['n_cases']):>{voxel_widths[1]},}"
            f"{int(row['n_inverted']):>{voxel_widths[2]},}"
            f"{_format_number(row['avg_voxel_volume_mm3'], 6):>{voxel_widths[3]}}"
            f"{_format_number(row['avg_voxel_count'], 0):>{voxel_widths[4]}}"
        )

    lines.append(voxel_sep)
    lines.append("")

    # Table 2: volume distribution information.
    distribution_widths = [group_width, 10, 10, 10, 16, 10, 10, 10, 10, 10, 10, 10, 10]
    distribution_sep    = "-" * sum(distribution_widths)

    lines.append("VOLUME DISTRIBUTION SUMMARY (cm³ unless otherwise noted)")
    lines.append(distribution_sep)
    lines.append(
        f"{'Group':<{distribution_widths[0]}}"
        f"{'Mean':>{distribution_widths[1]}}"
        f"{'Median':>{distribution_widths[2]}}"
        f"{'Std Dev':>{distribution_widths[3]}}"
        f"{'Variance (cm⁶)':>{distribution_widths[4]}}"
        f"{'Min':>{distribution_widths[5]}}"
        f"{'Q1':>{distribution_widths[6]}}"
        f"{'Q3':>{distribution_widths[7]}}"
        f"{'Max':>{distribution_widths[8]}}"
        f"{'IQR':>{distribution_widths[9]}}"
        f"{'Range':>{distribution_widths[10]}}"
        f"{'CV (%)':>{distribution_widths[11]}}"
        f"{'Skew':>{distribution_widths[12]}}"
    )
    lines.append(distribution_sep)

    for _, row in summary.iterrows():
        lines.append(
            f"{str(row['group']):<{distribution_widths[0]}}"
            f"{_format_number(row['mean_volume_cm3']):>{distribution_widths[1]}}"
            f"{_format_number(row['median_volume_cm3']):>{distribution_widths[2]}}"
            f"{_format_number(row['std_volume_cm3']):>{distribution_widths[3]}}"
            f"{_format_number(row['variance_volume_cm6']):>{distribution_widths[4]}}"
            f"{_format_number(row['min_volume_cm3']):>{distribution_widths[5]}}"
            f"{_format_number(row['q1_volume_cm3']):>{distribution_widths[6]}}"
            f"{_format_number(row['q3_volume_cm3']):>{distribution_widths[7]}}"
            f"{_format_number(row['max_volume_cm3']):>{distribution_widths[8]}}"
            f"{_format_number(row['iqr_volume_cm3']):>{distribution_widths[9]}}"
            f"{_format_number(row['range_volume_cm3']):>{distribution_widths[10]}}"
            f"{_format_number(row['cv_percent']):>{distribution_widths[11]}}"
            f"{_format_number(row['skewness'], 3):>{distribution_widths[12]}}"
        )

    lines.append(distribution_sep)

    return "\n".join(lines)


def extreme_case_paths(df, which="min", per_group=1) -> dict:
    """Pick the smallest- or largest-volume cases within each group.

    Args:
        df: Per-case DataFrame from :func:`collect_mask_statistics`.
        which: ``"min"`` for the smallest volumes, ``"max"`` for the largest.
        per_group: How many cases to return per group. ``0`` returns nothing.

    Returns:
        dict[str, list[str]]: Group name to the selected mask paths, ordered
        from most to least extreme.
    """
    if df is None or df.empty or per_group <= 0:
        return {}

    selected = {}

    for group_name, group_df in df.groupby("group", sort=False):
        if which == "max":
            rows = group_df.nlargest(per_group, "volume_cm3")

        else:
            rows = group_df.nsmallest(per_group, "volume_cm3")

        if not rows.empty:
            selected[group_name] = rows["path"].tolist()

    return selected
