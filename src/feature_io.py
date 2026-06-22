import numpy          as np
import pandas         as pd

from   pathlib        import Path

def save_features(features, save_path):
    np_features = {}

    for key, val in features.items():
        cleaned = [v if v is not None else np.zeros(1) for v in val]
        
        try:
            np_features[key] = np.array(cleaned)
        except ValueError:
            np_features[key] = np.array(cleaned, dtype=object)

    np.savez(save_path, **np_features)

def save_csv(features, filePaths, save_path):
    df         = pd.DataFrame(features)
    filenames  = [Path(p).name for p in filePaths]
    full_paths = [str(p) for p in filePaths]
    
    df.insert(0, "filepath", full_paths)
    df.insert(0, "filename", filenames)
    
    df.to_csv(save_path, index=False)

def load_features(npz_file_path):
    features = np.load(npz_file_path, allow_pickle=True)

    return {key: features[key] for key in features.files}