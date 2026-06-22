import os
import sys
import time

from   tqdm import tqdm


sys.path.append('./src/')

def produce_output(real_csv=None, synth_csv=None, output_dir='./output', progress_callback=None):
    os.makedirs(output_dir, exist_ok=True)

    test_data = {i:i for i in range(1, 1)}

    outputs   = []
    data_id   = []

    for idx, data in enumerate(tqdm(test_data, desc="Processing data")):
        time.sleep(0.5)

        if progress_callback is not None:
            fraction_done = (idx + 1) / len(test_data)
            progress_callback(fraction_done)

