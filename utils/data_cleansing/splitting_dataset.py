import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.extraction_log import logger as extraction_logger
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def split_dataset(data):
    test_ratio = 0.2
    total_size = data.shape[0]
    shuffled_indices = np.random.permutation(total_size)
    test_set_size = int(total_size * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

