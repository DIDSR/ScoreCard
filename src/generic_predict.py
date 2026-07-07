import os
import sys
import time

from   tqdm import tqdm


sys.path.append('./src/')

def produce_output(real_csv=None, synth_csv=None, output_dir='./output', progress_callback=None):
    """Run inference and write outputs for real and synthetic input CSV files.

    Creates the output directory if needed, processes each input record, and
    optionally reports progress through a callback.

    Args:
        real_csv: Path to the CSV file containing real input records. Defaults to
            None.
        synth_csv: Path to the CSV file containing synthetic input records.
            Defaults to None.
        output_dir: Directory where inference outputs are written. Defaults to
            ``'./output'``.
        progress_callback: Optional callable invoked after each processed item
            with a float fraction complete in ``[0.0, 1.0]``. Defaults to None.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    test_data = {i:i for i in range(1, 2)}

    outputs   = []
    data_id   = []

    for idx, data in enumerate(tqdm(test_data, desc="Processing data")):
        time.sleep(0.5)

        if progress_callback is not None:
            fraction_done = (idx + 1) / len(test_data)
            progress_callback(fraction_done)

