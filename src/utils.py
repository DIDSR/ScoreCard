import pydicom
import cv2
import random
import os

import numpy                 as np
import pandas                as pd

from   PIL                   import Image
from   src.feature_io        import load_features
from   src.routines          import combine_features, filter_features
from   src.compute_metrics   import (
    compute_congruence, 
    compute_coverage,
    compute_completeness,
    compute_consistency
)

def create_image_df(dir='./data/files') -> pd.DataFrame:
    """
    Scans a directory for DICOM (.dcm/.dicom) or PNG (.png) files and returns a DataFrame
    with their filenames and relative paths.

    Args:
        dir: Path to the directory to scan. Defaults to './data/files'.

    Returns:
        A DataFrame with columns:
            - image_name: The filename (e.g., 'scan_001.dcm')
            - image_path: The relative path to the file (e.g., './data/files/scan_001.dcm')
    """
    records = [
        {
            "image_name": f,
            "image_path": os.path.join(dir, f)
        }
        for f in os.listdir(dir)
        if f.lower().endswith(".dcm") or f.lower().endswith('.dicom') or f.lower().endswith('.png')
    ]

    df = pd.DataFrame(records, columns=["image_name", "image_path"])

    df.to_csv('./data/user_data.csv', index=False)

    return df


def generate_random_color():
    return "#{:06X}".format(random.randint(0, 0xFFFFFF))

def extend_colors(plot_colors, N):
    new_colors = set(plot_colors)
    
    while len(new_colors) < len(plot_colors) + N:
        new_colors.add(generate_random_color())
    
    return list(new_colors)

def read_png(png_fname):
    '''
    Reads a PNG image file and returns a preprocessed uint8 grayscale array.

    Processing steps applied:
        1. Read the image and convert to grayscale if needed
        2. Normalize to [0, 255] uint8

    Args:
        png_fname (str | Path): The filename or path to the PNG file to be read.

    Returns:
        np.ndarray: A uint8 NumPy array (0–255) of the preprocessed pixel data.
    '''
    img              = Image.open(str(png_fname))
    arr              = np.array(img, dtype=np.float32)

    arr_min, arr_max = arr.min(), arr.max()

    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0

    return arr.astype(np.uint8)


def hist_analysis(results_dir="./data/features"):
    from src.visualization import print_histograms

    real_features        = load_features(os.path.join(results_dir, 'real_patch_appearance_features.npz'))
    synth_features       = load_features(os.path.join(results_dir, 'static_patch_appearance_features.npz'))

    fig_hist             = print_histograms(real_features, synth_features)

    return fig_hist, None

def coverage_analysis(results_dir='./data/features'):
    from src.visualization import create_coverage_barplot

    real_features                    = os.path.join(results_dir, 'real_patch_appearance_features.npz')
    synth_features                   = os.path.join(results_dir, 'static_patch_appearance_features.npz')

    real_features_list               = []
    synth_features_list              = []

    real_features_list.append(real_features)
    synth_features_list.append(synth_features)

    real_df                          = combine_features(real_features_list)
    synth_df                         = combine_features(synth_features_list)

    real_df, synth_df, kept_features = filter_features(real_df, synth_df, 0.5)
    feature_names                    = [c for c in kept_features if c not in real_features]

    real_features_coverage           = compute_coverage(real_df[feature_names],  "Real")
    synth_features_coverage          = compute_coverage(synth_df[feature_names], "Synth")

    coverage_df                      = pd.DataFrame({
        'Real_Features':  real_features_coverage,
        'Synth_Features': synth_features_coverage,
    }).T

    fig_coverage = create_coverage_barplot(coverage_df)

    return fig_coverage


def congruence_analysis(results_dir='./data/features'):
    from src.visualization import create_congruence_barplot

    real_features        = os.path.join(results_dir, 'real_patch_appearance_features.npz')
    synth_features       = os.path.join(results_dir, 'static_patch_appearance_features.npz')

    real_features_list   = []
    synth_features_list  = []

    real_features_list.append(real_features)
    synth_features_list.append(synth_features)

    real_df              = combine_features(real_features_list)
    synth_df             = combine_features(synth_features_list)

    congruence_results = {}
    feature_names      = real_df.columns.tolist()

    for feature in feature_names:
        r                           = real_df[feature].values
        s                           = synth_df[feature].values
        congruence_results[feature] = compute_congruence(r, s, sampling=True, seed=42)

    summary_list = []

    for feature, values in congruence_results.items():
        summary_list.append({
            'Synthetic':         'Synth',
            'Real':              'Real',
            'Feature':           feature,
            'Cosine_Similarity': values['cosine_similarity'],
            'JSD':               values['jensen_shannon_divergence'],
            'EMD_Wasserstein':   values['earth_movers_distance']
        })

    congruence_df  = pd.DataFrame(summary_list)
    fig_congruence = create_congruence_barplot(congruence_df)

    return fig_congruence

def completeness_analysis(real_csv, synth_csv, required_fields=None, label=""):
    from src.visualization import create_completeness_barplot

    real_df         = pd.read_csv(real_csv)
    synth_df        = pd.read_csv(synth_csv)

    real_comp       = compute_completeness(real_df,  required_fields=required_fields, label=f"{label}_real")
    synth_comp      = compute_completeness(synth_df, required_fields=required_fields, label=f"{label}_synth")

    real_per_field  = real_comp.pop('per_field',  {})
    synth_per_field = synth_comp.pop('per_field', {})

    comp_df = pd.DataFrame([
        {'Dataset': 'Real',  **real_comp},
        {'Dataset': 'Synth', **synth_comp}
    ])

    try:
        fig = create_completeness_barplot(comp_df,
                                          real_per_field=real_per_field,    
                                          synth_per_field=synth_per_field)
    except Exception:
        fig = None

    return comp_df, fig


def consistency_analysis(csv_path, group_by="hospital", metric_cols=None, label=""):
    """
    Compute consistency metrics across one grouping column.
    The paper recommends evaluating one grouping variable at a time.
    """
    from src.visualization import create_consistency_barplot

    df = pd.read_csv(csv_path)

    if group_by not in df.columns:
        raise ValueError(f"group_by column '{group_by}' not found in CSV.")

    if metric_cols is None:
        default_metrics = ['mean_intensity', 'glcm_contrast', 'glcm_homogeneity',
                           'glcm_energy', 'glcm_correlation', 'mean_r', 'mean_g', 'mean_b']
        metric_cols = [c for c in default_metrics if c in df.columns]

    if not metric_cols:
        raise ValueError("No valid metric_cols found.")

    results = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue

        group_data = {
            str(g): group[metric].dropna().values
            for g, group in df.groupby(group_by)
            if len(group) > 0
        }

        if len(group_data) < 2:
            continue

        cons = compute_consistency(group_data, label=f"{label}_{metric}")
        results.append({'Group_By': group_by, 'Metric': metric, **cons})

    if not results:
        return pd.DataFrame(), None

    cons_df = pd.DataFrame(results)

    try:
        fig = create_consistency_barplot(cons_df, group_by=group_by)
    except Exception:
        fig = None

    return cons_df, fig