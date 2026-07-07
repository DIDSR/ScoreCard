import numpy          as np
import pandas         as pd

from   pathlib        import Path

def save_features(features, save_path):
    """Save a feature dictionary to a compressed NPZ archive.

    None values are replaced with zero-length arrays before serialization.

    Args:
        features: Dict mapping feature names to lists or arrays of values.
        save_path: Output path for the .npz file.
    """
    np_features = {}

    for key, val in features.items():
        cleaned = [v if v is not None else np.zeros(1) for v in val]
        
        try:
            np_features[key] = np.array(cleaned)
        except ValueError:
            np_features[key] = np.array(cleaned, dtype=object)

    np.savez(save_path, **np_features)

def save_csv(features, filePaths, save_path):
    """Write features and associated file metadata to a CSV file.

    Args:
        features: Dict or DataFrame-compatible mapping of feature columns.
        filePaths: Source file paths aligned with feature rows.
        save_path: Output path for the CSV file.
    """
    df         = pd.DataFrame(features)
    filenames  = [Path(p).name for p in filePaths]
    full_paths = [str(p) for p in filePaths]
    
    df.insert(0, "filepath", full_paths)
    df.insert(0, "filename", filenames)
    
    df.to_csv(save_path, index=False)

def load_features(npz_file_path):
    """Load feature arrays from a compressed NPZ archive.

    Args:
        npz_file_path: Path to the .npz feature file.

    Returns:
        dict: Feature names mapped to NumPy arrays.
    """
    features = np.load(npz_file_path, allow_pickle=True)

    return {key: features[key] for key in features.files}