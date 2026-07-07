import pydicom
import cv2
import random
import os

import numpy                 as np
import pandas                as pd

from   PIL                   import Image

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
    """Generate a random hex color string.

    Returns:
        str: Color in '#RRGGBB' format.
    """
    return "#{:06X}".format(random.randint(0, 0xFFFFFF))

def extend_colors(plot_colors, N):
    """Extend a color list with randomly generated unique colors.

    Args:
        plot_colors: Existing list of color strings.
        N: Number of additional unique colors to add.

    Returns:
        list: Original colors plus N new unique hex color strings.
    """
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

