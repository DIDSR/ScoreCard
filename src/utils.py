import pydicom
import cv2
import random
import os

import matplotlib.pyplot     as plt
import matplotlib.patches    as mpatches
import matplotlib.colors     as mcolors
import matplotlib.gridspec   as gridspec
import seaborn               as sns
import numpy                 as np
import pandas                as pd

from   PIL                   import Image
from   matplotlib.patches    import FancyArrowPatch
from   src.feature_io        import save_features, load_features, save_csv
from   src.routines          import combine_features, filter_features
from   src.compute_metrics   import compute_congruence, compute_coverage

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


def read_dicom(dcm_fname):
    '''
    Reads a DICOM mammogram file and returns a preprocessed uint8 pixel array.

    Processing steps applied:
        1. Read pixel data and apply RescaleSlope / RescaleIntercept
        2. Mask out padding pixels (Pixel Padding Value to Pixel Padding Range Limit)
        3. Apply VOI windowing (Window Center / Window Width) to clip to
           the diagnostically relevant intensity range
        4. Normalize the windowed, masked array to [0, 255] uint8
        5. Invert if PhotometricInterpretation is MONOCHROME1

    Args:
        dcm_fname (str | Path): The filename or path to the DICOM file to be read.

    Returns:
        np.ndarray: A uint8 NumPy array (0–255) of the preprocessed pixel data.
    '''
    # --- Load ---
    if str(dcm_fname).endswith('.gz'):
        with gzip.open(str(dcm_fname), 'rb') as f:
            dcm = pydicom.dcmread(f)
    else:
        dcm = pydicom.dcmread(str(dcm_fname))

    # --- Step 1: Rescale ---
    slope     = float(getattr(dcm, 'RescaleSlope',     1))
    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
    arr = dcm.pixel_array.astype(np.float32) * slope + intercept

    # --- Step 2: Padding mask ---
    # Pixels at or below the padding range are background (non-tissue).
    # They are excluded from windowing and set to the window's lower bound
    # so they map to 0 (black) after normalization.
    pad_value = float(getattr(dcm, 'PixelPaddingValue',          None) or -np.inf)
    pad_limit = float(getattr(dcm, 'PixelPaddingRangeLimit', pad_value))
    padding_mask = arr <= pad_limit  # True where pixel is background

    # --- Step 3: VOI Windowing ---
    # Clip pixel values to the diagnostically relevant window.
    # Pixels outside this range are saturated to black or white.
    if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
        wc = float(dcm.WindowCenter[0] if hasattr(dcm.WindowCenter, '__iter__')
                   else dcm.WindowCenter)
        ww = float(dcm.WindowWidth[0]  if hasattr(dcm.WindowWidth,  '__iter__')
                   else dcm.WindowWidth)
        win_low  = wc - ww / 2.0
        win_high = wc + ww / 2.0
    else:
        # Fall back to the data range if window tags are absent
        win_low  = float(np.min(arr[~padding_mask]))
        win_high = float(np.max(arr[~padding_mask]))

    arr = np.clip(arr, win_low, win_high)

    # Force padding pixels to the lower bound so they become 0 after scaling
    arr[padding_mask] = win_low

    # --- Step 4: Normalize to [0, 255] ---
    arr = (arr - win_low) / (win_high - win_low) * 255.0

    # --- Step 5: Photometric inversion ---
    # MONOCHROME1: high values = low intensity (must invert)
    # MONOCHROME2: high values = high intensity (no inversion needed)
    photometric = getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2').strip()
    if photometric == 'MONOCHROME1':
        arr = 255.0 - arr

    return arr.astype(np.uint8)

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
    img = Image.open(str(png_fname))
    arr = np.array(img, dtype=np.float32)

    # Normalize to [0, 255] only if the array is not already in that range
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

    real_features        = os.path.join(results_dir, 'real_patch_appearance_features.npz')
    synth_features       = os.path.join(results_dir, 'static_patch_appearance_features.npz')

    real_features_list   = []
    synth_features_list  = []

    real_features_list.append(real_features)
    synth_features_list.append(synth_features)

    real_df              = combine_features(real_features_list)
    synth_df             = combine_features(synth_features_list)

    real_df, synth_df, kept_features = filter_features(real_df, synth_df, 0.5)
    feature_names                    = [c for c in kept_features if c not in real_features]

    real_features_coverage  = compute_coverage(real_df[feature_names],  "Real")
    synth_features_coverage = compute_coverage(synth_df[feature_names], "Synth")

    coverage_df             = pd.DataFrame({
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