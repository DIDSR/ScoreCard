"""Brain CTA volume loading, texture feature extraction, and PCA.

This is the data layer for stage 2 of the brain pipeline (see
``src.brain_pipeline``). It walks an experiment tree of HDF5 CT volumes and
their paired masks, applies a brain window, and extracts Local Binary Pattern
and Gray-Level Co-occurrence Matrix texture features from the masked middle
axial slice of each case, then projects the feature matrix with PCA.

Expected layout::

    <experiment_root>/<condition>/<case>/ct_simulation/MIDA_deformed_<idx>.h5
    <experiment_root>/<condition>/<case>/ct_simulation/MIDA_deformed_<idx>.mask.h5

where ``<case>`` is named ``case_<idx>``, the CT file holds a ``(Z, Y, X)``
float32 dataset of Hounsfield Units, and the mask file holds a matching binary
``uint8`` dataset.

Nothing here draws figures; see ``src.brain_visualization``.
"""

import h5py

import numpy                 as     np
import pandas                as     pd

from   pathlib               import Path
from   skimage.feature       import local_binary_pattern, graycomatrix, graycoprops
from   sklearn.decomposition import PCA
from   sklearn.preprocessing import StandardScaler


DEFAULT_CT_DIR             = "ct_simulation"
DEFAULT_CT_DATASET         = "reconstruction"
DEFAULT_MASK_DATASET       = "masks/Blood Arteries"
DEFAULT_CT_FILE_TEMPLATE   = "MIDA_deformed_{index}.h5"
DEFAULT_MASK_FILE_TEMPLATE = "MIDA_deformed_{index}.mask.h5"

# Standard brain window, in Hounsfield Units.
BRAIN_WINDOW_CENTER = 40
BRAIN_WINDOW_WIDTH  = 80

DEFAULT_LBP_RADIUS = 3
DEFAULT_LBP_METHOD = "uniform"

DEFAULT_GLCM_DISTANCES = (1, 3, 5)
DEFAULT_GLCM_ANGLES    = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
DEFAULT_GLCM_PROPS     = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
)

GLCM_LEVELS = 256


def case_files(
    root,
    condition,
    case,
    ct_dir=DEFAULT_CT_DIR,
    ct_file_template=DEFAULT_CT_FILE_TEMPLATE,
    mask_file_template=DEFAULT_MASK_FILE_TEMPLATE,
):
    """Derive the CT and mask paths for one case folder.

    Args:
        root: Experiment root directory.
        condition: Condition (acquisition setting) folder name.
        case: Case folder name, such as ``case_0000``; the trailing token is
            used as the zero-padded file index.
        ct_dir: Sub-directory holding the simulated CT files.
        ct_file_template: Template for the CT filename, formatted with
            ``index``.
        mask_file_template: Template for the mask filename, formatted with
            ``index``.

    Returns:
        tuple[pathlib.Path, pathlib.Path]: ``(ct_path, mask_path)``.
    """
    index = str(case).split("_")[-1]
    base  = Path(root) / condition / case / ct_dir

    return (
        base / ct_file_template.format(index=index),
        base / mask_file_template.format(index=index),
    )


def read_h5_dataset(path, dataset_key) -> np.ndarray:
    """Read one dataset out of an HDF5 file.

    Args:
        path: Path to the HDF5 file.
        dataset_key: Dataset key, such as ``reconstruction`` or
            ``masks/Blood Arteries``.

    Returns:
        numpy.ndarray: The dataset contents.

    Raises:
        KeyError: When the file does not contain ``dataset_key``.
    """
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise KeyError(f"{path} has no dataset {dataset_key!r}")

        return handle[dataset_key][:]


def load_case_volumes(
    ct_path,
    mask_path,
    ct_dataset=DEFAULT_CT_DATASET,
    mask_dataset=DEFAULT_MASK_DATASET,
):
    """Load the CT volume and its paired mask volume.

    Args:
        ct_path: Path to the CT HDF5 file.
        mask_path: Path to the mask HDF5 file.
        ct_dataset: Dataset key inside the CT file.
        mask_dataset: Dataset key inside the mask file.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(ct, mask)``, both ``(Z, Y, X)``.
    """
    return (
        read_h5_dataset(ct_path, ct_dataset),
        read_h5_dataset(mask_path, mask_dataset),
    )


def apply_window(volume, center=BRAIN_WINDOW_CENTER, width=BRAIN_WINDOW_WIDTH):
    """Clip CT Hounsfield Units to a display window.

    Args:
        volume: CT array in Hounsfield Units.
        center: Window centre, in HU. Defaults to the brain window.
        width: Window width, in HU. Defaults to the brain window.

    Returns:
        numpy.ndarray: Volume clipped to ``[center - width/2, center + width/2]``.
    """
    low  = center - width / 2.0
    high = center + width / 2.0

    return np.clip(volume, low, high)


def normalize_to_uint8(volume) -> np.ndarray:
    """Scale a windowed volume to the ``[0, 255]`` range texture analysis needs.

    Args:
        volume: Windowed CT array.

    Returns:
        numpy.ndarray: ``uint8`` array. A constant-valued volume maps to zeros.
    """
    volume = np.asarray(volume, dtype=np.float64)

    minimum = float(volume.min())
    maximum = float(volume.max())

    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return np.zeros(volume.shape, dtype=np.uint8)

    scaled = (volume - minimum) / (maximum - minimum) * 255.0

    return scaled.astype(np.uint8)


def get_middle_slice(volume) -> np.ndarray:
    """Return the middle axial slice of a ``(Z, Y, X)`` volume."""
    return volume[volume.shape[0] // 2]


def lbp_bin_count(n_points, method=DEFAULT_LBP_METHOD) -> int:
    """Return the fixed number of LBP codes produced by a method.

    Binning on a fixed count keeps every case's histogram the same length. A
    count derived from the observed maximum instead varies between slices and
    yields a ragged feature matrix.

    Args:
        n_points: Number of circular sampling points.
        method: ``local_binary_pattern`` method name.

    Returns:
        int: Number of possible LBP codes.

    Raises:
        ValueError: For methods without a fixed code count, such as ``var``.
    """
    method = str(method).lower()

    if method == "uniform":
        return int(n_points) + 2

    if method == "nri_uniform":
        return int(n_points) * (int(n_points) - 1) + 3

    if method in ("default", "ror"):
        return 2 ** int(n_points)

    raise ValueError(
        f"LBP method {method!r} has no fixed histogram length; "
        "use 'uniform', 'nri_uniform', 'default', or 'ror'."
    )


def extract_lbp_histogram(
    gray_slice,
    mask_slice,
    radius=DEFAULT_LBP_RADIUS,
    n_points=None,
    method=DEFAULT_LBP_METHOD,
) -> np.ndarray:
    """Compute a normalized LBP histogram over the masked pixels of a slice.

    Args:
        gray_slice: 2-D ``uint8`` image.
        mask_slice: 2-D mask; pixels greater than zero are included.
        radius: LBP radius, in pixels.
        n_points: Number of sampling points. Defaults to ``8 * radius``.
        method: ``local_binary_pattern`` method name.

    Returns:
        numpy.ndarray: Density histogram of length
        ``lbp_bin_count(n_points, method)``.
    """
    n_points = int(n_points if n_points is not None else 8 * radius)
    n_bins   = lbp_bin_count(n_points, method)

    lbp       = local_binary_pattern(gray_slice, n_points, radius, method)
    histogram = np.histogram(
        lbp[np.asarray(mask_slice) > 0],
        bins=n_bins,
        range=(0, n_bins),
        density=True,
    )[0]

    return histogram


def extract_glcm_vector(
    gray_slice,
    mask_slice,
    distances=DEFAULT_GLCM_DISTANCES,
    angles=DEFAULT_GLCM_ANGLES,
    props=DEFAULT_GLCM_PROPS,
    levels=GLCM_LEVELS,
) -> np.ndarray:
    """Compute GLCM property means over the mask bounding box of a slice.

    Args:
        gray_slice: 2-D ``uint8`` image.
        mask_slice: 2-D mask used to crop the region of interest.
        distances: GLCM offsets, in pixels.
        angles: GLCM angles, in radians.
        props: Properties passed to ``graycoprops``.
        levels: Gray levels in the co-occurrence matrix.

    Returns:
        numpy.ndarray: One mean value per property, averaged over all
        distance/angle pairs.
    """
    mask_slice = np.asarray(mask_slice)

    rows = np.any(mask_slice > 0, axis=1)
    cols = np.any(mask_slice > 0, axis=0)

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    roi  = gray_slice[r0:r1 + 1, c0:c1 + 1]
    glcm = graycomatrix(
        roi,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )

    return np.array([float(graycoprops(glcm, prop).mean()) for prop in props])


def texture_feature_names(
    radius=DEFAULT_LBP_RADIUS,
    n_points=None,
    method=DEFAULT_LBP_METHOD,
    props=DEFAULT_GLCM_PROPS,
) -> list:
    """Name every column of the combined texture feature vector.

    Args:
        radius: LBP radius, in pixels.
        n_points: Number of sampling points. Defaults to ``8 * radius``.
        method: ``local_binary_pattern`` method name.
        props: GLCM properties.

    Returns:
        list[str]: ``lbp_00 … lbp_NN`` followed by ``glcm_<prop>`` names.
    """
    n_points = int(n_points if n_points is not None else 8 * radius)
    n_bins   = lbp_bin_count(n_points, method)

    return (
        [f"lbp_{index:02d}" for index in range(n_bins)]
        + [f"glcm_{str(prop).lower()}" for prop in props]
    )


def extract_case_features(
    ct_volume,
    mask_volume,
    window_center=BRAIN_WINDOW_CENTER,
    window_width=BRAIN_WINDOW_WIDTH,
    lbp_radius=DEFAULT_LBP_RADIUS,
    lbp_n_points=None,
    lbp_method=DEFAULT_LBP_METHOD,
    glcm_distances=DEFAULT_GLCM_DISTANCES,
    glcm_angles=DEFAULT_GLCM_ANGLES,
    glcm_props=DEFAULT_GLCM_PROPS,
):
    """Extract the combined LBP + GLCM vector for one case.

    The CT volume is windowed, normalized to ``uint8``, and reduced to its
    middle axial slice before features are taken over the masked region.

    Args:
        ct_volume: ``(Z, Y, X)`` CT array in Hounsfield Units.
        mask_volume: ``(Z, Y, X)`` binary mask array.
        window_center: CT window centre, in HU.
        window_width: CT window width, in HU.
        lbp_radius: LBP radius, in pixels.
        lbp_n_points: LBP sampling points. Defaults to ``8 * lbp_radius``.
        lbp_method: ``local_binary_pattern`` method name.
        glcm_distances: GLCM offsets, in pixels.
        glcm_angles: GLCM angles, in radians.
        glcm_props: GLCM properties to average.

    Returns:
        numpy.ndarray | None: The feature vector, or ``None`` when the middle
        axial slice of the mask is empty.
    """
    ct_slice   = get_middle_slice(
        normalize_to_uint8(apply_window(ct_volume, window_center, window_width))
    )
    mask_slice = get_middle_slice(mask_volume)

    if np.count_nonzero(mask_slice) == 0:
        return None

    lbp_histogram = extract_lbp_histogram(
        ct_slice,
        mask_slice,
        radius=lbp_radius,
        n_points=lbp_n_points,
        method=lbp_method,
    )

    glcm_vector = extract_glcm_vector(
        ct_slice,
        mask_slice,
        distances=glcm_distances,
        angles=glcm_angles,
        props=glcm_props,
    )

    return np.concatenate([lbp_histogram, glcm_vector])


def discover_cases(
    experiment_root,
    conditions=None,
    max_cases_per_condition=None,
) -> list:
    """List the ``(condition, case)`` folders under an experiment root.

    Args:
        experiment_root: Directory containing one sub-directory per condition.
        conditions: Optional subset of condition names to keep. ``None`` uses
            every condition found.
        max_cases_per_condition: Optional cap on cases per condition, applied
            after sorting, which is useful for a quick trial run.

    Returns:
        list[tuple[str, str]]: Sorted ``(condition, case)`` pairs.

    Raises:
        FileNotFoundError: When ``experiment_root`` does not exist.
    """
    root = Path(experiment_root)

    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")

    available = sorted(path.name for path in root.iterdir() if path.is_dir())

    if conditions is not None:
        wanted    = [str(condition) for condition in conditions]
        available = [name for name in available if name in wanted]

    pairs = []

    for condition in available:
        cases = sorted(
            path.name
            for path in (root / condition).iterdir()
            if path.is_dir()
        )

        if max_cases_per_condition is not None:
            cases = cases[:max_cases_per_condition]

        pairs.extend((condition, case) for case in cases)

    return pairs


def build_feature_table(
    experiment_root,
    conditions=None,
    max_cases_per_condition=None,
    ct_dir=DEFAULT_CT_DIR,
    ct_dataset=DEFAULT_CT_DATASET,
    mask_dataset=DEFAULT_MASK_DATASET,
    ct_file_template=DEFAULT_CT_FILE_TEMPLATE,
    mask_file_template=DEFAULT_MASK_FILE_TEMPLATE,
    window_center=BRAIN_WINDOW_CENTER,
    window_width=BRAIN_WINDOW_WIDTH,
    lbp_radius=DEFAULT_LBP_RADIUS,
    lbp_n_points=None,
    lbp_method=DEFAULT_LBP_METHOD,
    glcm_distances=DEFAULT_GLCM_DISTANCES,
    glcm_angles=DEFAULT_GLCM_ANGLES,
    glcm_props=DEFAULT_GLCM_PROPS,
    verbose=True,
):
    """Extract texture features for every case under an experiment root.

    Cases are skipped when either HDF5 file is missing, when the mask is empty
    on the middle axial slice, or when reading fails.

    Args:
        experiment_root: Directory containing one sub-directory per condition.
        conditions: Optional subset of condition names.
        max_cases_per_condition: Optional cap on cases per condition.
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
        verbose: When True, print a per-condition progress line and a summary.

    Returns:
        tuple: ``(features_df, skipped)`` where ``features_df`` has ``condition``
        and ``case`` columns followed by one column per texture feature, and
        ``skipped`` is a list of ``(condition, case, reason)`` triples.
    """
    feature_names = texture_feature_names(
        radius=lbp_radius,
        n_points=lbp_n_points,
        method=lbp_method,
        props=glcm_props,
    )

    pairs   = discover_cases(experiment_root, conditions, max_cases_per_condition)
    rows    = []
    skipped = []

    if verbose:
        n_conditions = len({condition for condition, _ in pairs})
        print(f"Found {len(pairs)} case(s) across {n_conditions} condition(s).")

    current_condition = None

    for condition, case in pairs:
        if verbose and condition != current_condition:
            current_condition = condition
            print(f"  {condition}")

        ct_path, mask_path = case_files(
            experiment_root,
            condition,
            case,
            ct_dir=ct_dir,
            ct_file_template=ct_file_template,
            mask_file_template=mask_file_template,
        )

        if not ct_path.exists() or not mask_path.exists():
            skipped.append((condition, case, "missing CT or mask file"))

            continue

        try:
            ct_volume, mask_volume = load_case_volumes(
                ct_path,
                mask_path,
                ct_dataset=ct_dataset,
                mask_dataset=mask_dataset,
            )

            features = extract_case_features(
                ct_volume,
                mask_volume,
                window_center=window_center,
                window_width=window_width,
                lbp_radius=lbp_radius,
                lbp_n_points=lbp_n_points,
                lbp_method=lbp_method,
                glcm_distances=glcm_distances,
                glcm_angles=glcm_angles,
                glcm_props=glcm_props,
            )

        except Exception as exc:
            skipped.append((condition, case, str(exc)))

            if verbose:
                print(f"    [ERROR] {case} — {exc}")

            continue

        if features is None:
            skipped.append((condition, case, "empty mask on middle slice"))

            if verbose:
                print(f"    [SKIP]  {case} — empty mask on middle slice")

            continue

        rows.append(
            {"condition": condition, "case": case}
            | dict(zip(feature_names, features.tolist()))
        )

    features_df = pd.DataFrame(rows, columns=["condition", "case"] + feature_names)

    if verbose:
        print(
            f"\nFeature table: {len(features_df)} sample(s) × "
            f"{len(feature_names)} feature(s); {len(skipped)} skipped."
        )

    return features_df, skipped


def feature_columns(features_df) -> list:
    """Return the texture feature column names of a feature table."""
    return [
        column
        for column in features_df.columns
        if column not in ("condition", "case")
    ]


def pca_projection(features_df, n_components=2) -> dict:
    """Standardize the texture features and project them with PCA.

    Args:
        features_df: Feature table from :func:`build_feature_table`.
        n_components: Number of principal components to keep.

    Returns:
        dict: Keys ``components`` (the projected array), ``explained_variance_ratio``,
        ``feature_names``, ``scaler``, and ``pca``.

    Raises:
        ValueError: When the table holds fewer samples than requested
            components.
    """
    names  = feature_columns(features_df)
    matrix = features_df[names].to_numpy(dtype=float)

    if matrix.shape[0] < n_components:
        raise ValueError(
            f"PCA needs at least {n_components} samples; got {matrix.shape[0]}."
        )

    scaler   = StandardScaler()
    embedded = PCA(n_components=n_components)

    components = embedded.fit_transform(scaler.fit_transform(matrix))

    return {
        "components"              : components,
        "explained_variance_ratio": embedded.explained_variance_ratio_,
        "feature_names"           : names,
        "scaler"                  : scaler,
        "pca"                     : embedded,
    }
