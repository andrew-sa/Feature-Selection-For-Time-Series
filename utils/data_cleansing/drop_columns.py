import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.extraction_log import logger as extraction_logger
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def drop_columns_with_nan_values(data):
    extraction_logger.info('Dataset colums: {0}'.format(data.shape[1]), extra = LOGGER_EXTRA_OBJECT)
    data = data.dropna(axis=1)
    extraction_logger.info('Dataset colums without NaN: {0}'.format(data.shape[1]), extra = LOGGER_EXTRA_OBJECT)
    return data