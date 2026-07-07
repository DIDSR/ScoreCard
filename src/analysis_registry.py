from __future__ import annotations

import os
import numpy          as     np
import pandas         as     pd


from   copy           import deepcopy
from   src.feature_io import load_features

PATCH_SKIP_COLS    = {"filename", "filepath", "image_name", "image_path"}
METADATA_SKIP_COLS = PATCH_SKIP_COLS | {"index", "Unnamed: 0"}
FEATURES_DIR       = "./data/features"
REAL_NPZ           = "real_patch_appearance_features.npz"
SYNTH_NPZ          = "kde_patch_appearance_features.npz"

ANALYSIS_REGISTRY = [
    {
        "id"         : "congruence",
        "label"      : "Congruence",
        "step_class" : "step-congruence",
        "data_source": "patch_npz",
        "description": "Statistical similarity between real and synthetic patch feature distributions.",
        "metrics"    : [
                        {"id": "jsd", "label": "Jensen-Shannon Divergence"},
                        {"id": "emd", "label": "Earth Mover's Distance"},
                        {"id": "cosine", "label": "Cosine Similarity"},
                       ],
        "features_label": "Patch features",
    },
    {
        "id"         : "coverage",
        "label"      : "Coverage",
        "step_class" : "step-coverage",
        "data_source": "patch_npz",
        "description": "Extent to which synthetic data covers the real feature space.",
        "metrics"    : [
                        {"id": "Variance", "label": "Variance"},
                        {"id": "Entropy", "label": "Entropy"},
                        {"id": "Distance_to_Centroid", "label": "Distance to Centroid"},
                        {"id": "Convex_Hull_Volume", "label": "Convex Hull Volume"},
                       ],
        "features_label": "Patch features",
    },
    {
        "id": "completeness",
        "label": "Completeness",
        "step_class": "step-completeness",
        "data_source": "metadata_csv",
        "description": "Missing values and required-field fill rates in metadata CSVs.",
        "metrics": [
            {"id": "Missing_Data_Percentage", "label": "Missing Data Percentage"},
            {"id": "Required_Fields_Completeness", "label": "Required Fields Completeness"},
            {"id": "Per_Field", "label": "Per-Field Breakdown"},
        ],
        "features_label": "Metadata fields",
    },
    {
        "id": "consistency",
        "label": "Consistency",
        "step_class": "step-consistency",
        "data_source": "metadata_csv",
        "description": "Stability of numeric metadata across demographic subgroups.",
        "metrics": [
            {"id": "Variance_of_Group_Means", "label": "Variance of Group Means"},
            {"id": "Max_Min_Difference", "label": "Max-Min Difference"},
            {"id": "ANOVA_F_statistic", "label": "ANOVA F-Statistic"},
        ],
        "features_label": "Numeric metadata fields",
        "has_group_by": True,
    },
    {
        "id": "histogram",
        "label": "Histograms",
        "step_class": "step-histogram",
        "data_source": "patch_npz",
        "description": "Overlaid feature distribution histograms for real vs. synthetic data.",
        "metrics": [],
        "features_label": "Patch features",
    },
    {
        "id": "constraint",
        "label": "Constraint",
        "step_class": "step-constraint",
        "data_source": "patch_npz",
        "description": "Rate at which synthetic values violate real-data percentile bounds.",
        "metrics": [
            {"id": "Synth_Violation_%", "label": "Violation Rate"},
        ],
        "features_label": "Patch features",
    },
]

CONSISTENCY_DEFAULT_FIELDS = [
    "Age at dx",
    "BMI at dx (kg)",
    "BMI at follow-up (kg)",
    "mpp",
    "compressionratio",
    "exposure time",
]

COMPLETENESS_PREFERRED_FIELDS = [
    "Patient ID",
    "Age at dx",
    "BMI at dx (kg)",
    "BMI at follow-up (kg)",
    "Race",
    "DM (Y/N)",
    "Progestin Use (type/agent)",
    "FHx of endometrial CA",
    "PHx of breast/ovarian CA",
    "vendor",
    "Sex",
    "Ethnicity",
    "Initial dx",
    "Responder?",
]


def _load_patch_feature_names(features_dir=FEATURES_DIR):
    """Load sorted patch feature names shared by real and synthetic NPZ files.

    Args:
        features_dir: Directory containing real and synthetic patch feature NPZ
            files.

    Returns:
        A sorted list of feature names present in both NPZ files when both exist;
        otherwise names from whichever file is available.
    """
    real_path = os.path.join(features_dir, REAL_NPZ)
    synth_path = os.path.join(features_dir, SYNTH_NPZ)
    names = []

    if os.path.exists(real_path):
        real_feats = load_features(real_path)
        names = sorted(real_feats.keys())

    if os.path.exists(synth_path):
        synth_feats = load_features(synth_path)
        synth_keys = set(synth_feats.keys())
        if names:
            names = [n for n in names if n in synth_keys]
        else:
            names = sorted(synth_keys)

    return names


def _metadata_columns(real_csv, synth_csv, patch_features=None):
    """Identify shared metadata columns excluding patch and identifier fields.

    Args:
        real_csv: Path to the real-data CSV used to infer column names.
        synth_csv: Path to the synthetic-data CSV used to infer column names.
        patch_features: Optional list of patch feature column names to exclude
            from the metadata column set.

    Returns:
        A sorted list of column names present in both CSVs, excluding skip columns
        and any patch feature names.
    """
    real_df = pd.read_csv(real_csv, nrows=500)
    synth_df = pd.read_csv(synth_csv, nrows=500)
    real_df.columns = real_df.columns.str.strip()
    synth_df.columns = synth_df.columns.str.strip()
    shared = sorted(set(real_df.columns) & set(synth_df.columns))

    patch_cols = set(patch_features or [])
    return [
        c for c in shared
        if c not in METADATA_SKIP_COLS and c not in patch_cols
    ]


def _categorical_columns(real_csv, synth_csv, metadata_columns):
    """Detect low-cardinality categorical columns suitable for grouping.

    Args:
        real_csv: Path to the real-data CSV.
        synth_csv: Path to the synthetic-data CSV.
        metadata_columns: Candidate metadata column names to evaluate.

    Returns:
        A list of column names with between 2 and 25 unique non-null values across
        both datasets, ordered with preferred demographic columns first.
    """
    real_df = pd.read_csv(real_csv, usecols=lambda c: c in metadata_columns, nrows=1000)
    synth_df = pd.read_csv(synth_csv, usecols=lambda c: c in metadata_columns, nrows=1000)
    cats = []

    for col in metadata_columns:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        combined = pd.concat([real_df[col], synth_df[col]], ignore_index=True)
        combined = combined.replace(r"^\s*$", np.nan, regex=True).dropna()
        if combined.empty:
            continue
        nunique = combined.nunique()
        if 1 < nunique <= 25:
            cats.append(col)

    preferred = ["Race", "vendor", "Group", "Sex", "Ethnicity"]
    cats = sorted(cats, key=lambda c: (preferred.index(c) if c in preferred else 99, c))
    return cats


def _numeric_columns(real_csv, synth_csv, metadata_columns):
    """Collect numeric metadata columns present in both CSVs.

    Args:
        real_csv: Path to the real-data CSV.
        synth_csv: Path to the synthetic-data CSV.
        metadata_columns: Candidate metadata column names to evaluate.

    Returns:
        A list of numeric column names, with default consistency fields listed
        first followed by any additional numeric columns.
    """
    real_df = pd.read_csv(real_csv, nrows=500)
    synth_df = pd.read_csv(synth_csv, nrows=500)
    nums = []

    for col in metadata_columns:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(real_df[col]) and pd.api.types.is_numeric_dtype(synth_df[col]):
            nums.append(col)

    default = [c for c in CONSISTENCY_DEFAULT_FIELDS if c in nums]
    extras = [c for c in nums if c not in default]
    return default + extras


def discover_schema(
    real_csv,
    synth_csv,
    features_dir=FEATURES_DIR,
    metadata_real_csv=None,
    metadata_synth_csv=None,
):
    """Discover available patch features and metadata schema from input files.

    Args:
        real_csv: Path to the real-data CSV (image list or metadata).
        synth_csv: Path to the synthetic-data CSV.
        features_dir: Directory containing patch feature NPZ files.
        metadata_real_csv: Optional separate real metadata CSV; defaults to
            ``real_csv``.
        metadata_synth_csv: Optional separate synthetic metadata CSV; defaults to
            ``synth_csv``.

    Returns:
        A dict with keys ``patch_features``, ``metadata_columns``,
        ``categorical_columns``, ``numeric_columns``, resolved metadata CSV paths,
        and ``warnings`` describing schema issues.
    """
    patch_features = _load_patch_feature_names(features_dir)

    meta_real = metadata_real_csv or real_csv
    meta_synth = metadata_synth_csv or synth_csv

    metadata_columns = _metadata_columns(meta_real, meta_synth, patch_features)
    categorical_columns = _categorical_columns(meta_real, meta_synth, metadata_columns)
    numeric_columns = _numeric_columns(meta_real, meta_synth, metadata_columns)

    warnings = []
    using_separate_metadata = (
        metadata_real_csv is not None
        and metadata_synth_csv is not None
        and (metadata_real_csv != real_csv or metadata_synth_csv != synth_csv)
    )

    if not metadata_columns:
        warnings.append(
            "No metadata columns found in the loaded CSVs. Completeness and Consistency "
            "require patient or study metadata columns — not patch ML features "
            "(e.g. mean_intensity, glcm_contrast). Upload metadata-rich CSVs or use the "
            "preset metadata files."
        )
    elif not using_separate_metadata and patch_features:
        patch_overlap = [c for c in metadata_columns if c in patch_features]
        if patch_overlap:
            warnings.append(
                "Some columns look like patch ML features and were excluded from metadata analyses."
            )

    if metadata_columns and not categorical_columns:
        warnings.append(
            "No categorical metadata columns found for Consistency. "
            "Choose a group-by column such as Race or vendor."
        )
    elif metadata_columns and "Race" not in categorical_columns:
        warnings.append(
            "Column 'Race' was not found. Consistency will use another categorical column if available."
        )

    return {
        "patch_features": patch_features,
        "metadata_columns": metadata_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "metadata_real_csv": meta_real,
        "metadata_synth_csv": meta_synth,
        "warnings": warnings,
    }


def default_config(schema):
    """Build a default analysis configuration from a discovered schema.

    Args:
        schema: Schema dict returned by ``discover_schema``, containing patch
            features, metadata columns, and column type lists.

    Returns:
        A nested dict keyed by analysis id, with enabled flags, selected metrics,
        features or fields, and consistency group-by defaults.
    """
    patch = schema.get("patch_features") or []
    metadata = schema.get("metadata_columns") or []
    numeric = schema.get("numeric_columns") or []
    cats = schema.get("categorical_columns") or []

    completeness_fields = [c for c in COMPLETENESS_PREFERRED_FIELDS if c in metadata]
    if not completeness_fields:
        completeness_fields = metadata[:12]

    consistency_fields = [c for c in CONSISTENCY_DEFAULT_FIELDS if c in numeric]
    if not consistency_fields:
        consistency_fields = numeric[:6]

    group_by = "Race" if "Race" in cats else (cats[0] if cats else "")

    return {
        "congruence": {
            "enabled": True,
            "metrics": {"jsd": True, "emd": True, "cosine": True},
            "features": list(patch),
        },
        "coverage": {
            "enabled": True,
            "metrics": {
                "Variance": True,
                "Entropy": True,
                "Distance_to_Centroid": True,
                "Convex_Hull_Volume": True,
            },
            "features": list(patch),
        },
        "completeness": {
            "enabled": True,
            "metrics": {
                "Missing_Data_Percentage": True,
                "Required_Fields_Completeness": True,
                "Per_Field": True,
            },
            "fields": list(completeness_fields),
        },
        "consistency": {
            "enabled": True,
            "metrics": {
                "Variance_of_Group_Means": True,
                "Max_Min_Difference": True,
                "ANOVA_F_statistic": True,
            },
            "group_by": group_by,
            "fields": list(consistency_fields),
        },
        "histogram": {
            "enabled": True,
            "features": list(patch),
        },
        "constraint": {
            "enabled": True,
            "metrics": {"Synth_Violation_%": True},
            "features": list(patch),
        },
    }


def enabled_flags(analysis_config):
    """Extract enabled/disabled state for each registered analysis.

    Args:
        analysis_config: Full analysis configuration dict keyed by analysis id.

    Returns:
        A dict mapping each analysis id to its ``enabled`` boolean value.
    """
    return {aid: analysis_config.get(aid, {}).get("enabled", False) for aid in _analysis_ids()}


def _analysis_ids():
    """Return the ordered list of registered analysis identifiers.

    Returns:
        A list of analysis id strings defined in ``ANALYSIS_REGISTRY``.
    """
    return [entry["id"] for entry in ANALYSIS_REGISTRY]


def validate_config(config, schema):
    """Validate an analysis configuration against schema and registry rules.

    Args:
        config: Analysis configuration dict keyed by analysis id.
        schema: Schema dict returned by ``discover_schema``.

    Returns:
        A tuple ``(validated, errors)`` where ``validated`` is a deep copy of
        ``config`` and ``errors`` is a list of human-readable validation messages.
    """
    errors = []
    validated = deepcopy(config)

    for entry in ANALYSIS_REGISTRY:
        aid = entry["id"]
        block = validated.get(aid, {})
        if not block.get("enabled"):
            continue

        if entry["metrics"]:
            metrics = block.get("metrics", {})
            if not any(metrics.values()):
                errors.append(f"{entry['label']}: select at least one metric.")

        if entry["data_source"] == "patch_npz":
            features = block.get("features") or []
            if not features:
                errors.append(f"{entry['label']}: select at least one patch feature.")

        if entry["data_source"] == "metadata_csv":
            fields = block.get("fields") or []
            if not fields and aid != "completeness":
                errors.append(f"{entry['label']}: select at least one metadata field.")
            if aid == "completeness" and not fields:
                errors.append("Completeness: select at least one metadata field.")

        if entry.get("has_group_by"):
            group_by = block.get("group_by", "")
            if not group_by:
                errors.append("Consistency: choose a group-by column.")
            elif group_by not in schema.get("categorical_columns", []):
                errors.append(f"Consistency: group-by column '{group_by}' is not available.")

    return validated, errors


def parse_config_from_form(form, schema, existing_config=None):
    """Build analysis_config from POSTed configure form."""
    base = deepcopy(existing_config) if existing_config else default_config(schema)
    parsed = {}

    for entry in ANALYSIS_REGISTRY:
        aid = entry["id"]
        block = deepcopy(base.get(aid, {}))
        block["enabled"] = form.get(f"{aid}_enabled") == "on"

        if entry["metrics"]:
            selected = set(form.getlist(f"{aid}_metrics"))
            block["metrics"] = {
                m["id"]: (m["id"] in selected) for m in entry["metrics"]
            }

        if entry["data_source"] == "patch_npz":
            selected_features = form.getlist(f"{aid}_features")
            block["features"] = selected_features or block.get("features", [])

        if entry["data_source"] == "metadata_csv":
            selected_fields = form.getlist(f"{aid}_fields")
            block["fields"] = selected_fields or block.get("fields", [])

        if entry.get("has_group_by"):
            block["group_by"] = form.get(f"{aid}_group_by") or block.get("group_by", "")

        parsed[aid] = block

    return validate_config(parsed, schema)


def feature_options_for_analysis(analysis_id, schema):
    """Return selectable feature or field names for a given analysis type.

    Args:
        analysis_id: Identifier of the analysis (e.g. ``congruence``,
            ``completeness``).
        schema: Schema dict returned by ``discover_schema``.

    Returns:
        A list of column or feature names appropriate for the analysis data source;
        empty when ``analysis_id`` is not registered.
    """
    entry = next((e for e in ANALYSIS_REGISTRY if e["id"] == analysis_id), None)
    if not entry:
        return []

    if entry["data_source"] == "patch_npz":
        return schema.get("patch_features", [])
    if analysis_id == "consistency":
        return schema.get("numeric_columns", [])
    return schema.get("metadata_columns", [])