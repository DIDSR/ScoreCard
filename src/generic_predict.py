import os
import sys
import shutil
import gc
import time
import yaml
import gzip
import glob
import pydicom
import cv2
import logging

import numpy           as np
import pandas          as pd

from   PIL             import Image
from   pathlib         import Path
from   tqdm            import tqdm
from   datetime        import datetime
from   sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, roc_auc_score


sys.path.append('./src/')


def produce_output(real_csv=None, synth_csv=None, output_dir='./output', progress_callback=None):
    os.makedirs(output_dir, exist_ok=True)

    # DATA SETUP
    test_data = {i:i for i in range(1,10)} # Something you can iterate over. Could be a data loader. 

    # MAIN PROCESSING / inference loop / separate function call

    outputs = []
    data_id = []

    for idx, data in enumerate(tqdm(test_data, desc="Processing data")):

        # data_id.append(idx)
        # output = model(data)
        # outputs.append(output)

        time.sleep(0.5)

        if progress_callback is not None:
            fraction_done = (idx + 1) / len(test_data)
            progress_callback(fraction_done)
    
    # outputs = np.vstack(outputs)

    # results = {
    #     'data_id': data_id,
    #     'output': outputs,
    # }

    # # Additional processing on results if necessary

    # # Save results
    # df = pd.DataFrame(results)
    
    # # Save CSV
    # csv_path = os.path.join(output_dir, 'results.csv')
    # df.to_csv(csv_path, index=False)

