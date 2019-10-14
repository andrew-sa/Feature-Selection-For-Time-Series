import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.extraction_log import extraction_logger
# from utils.feature_extraction import feature_extractor
from utils.data_cleansing import rebalance_train_and_test_sets
import pandas as pd
import time

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

if __name__ == '__main__':

    datasets = ['ECG200', 'ECG5000', 'FordA', 'FordB', 'ChlorineConcentration', 'PhalangesOutlinesCorrect',
                'RefrigerationDevices', 'TwoLeadECG', 'TwoPatterns']

    # Extract features from all datasets 
    for dataset in datasets:
        # rebalance_train_and_test_sets.rebalance(dataset)
        extraction_logger.info('START EXTRACTING {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)
        feature_extractor.extract_dataset_features(dataset)
        extraction_logger.info('END EXTRACTING {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)
        time.sleep(120)

    # Or extract features from a particular dataset
    # dataset = datasets[0] #Specify the dataset
    # logger.info('START EXTRACTING {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)
    # feature_extractor.extract_dataset_features(dataset)
    # logger.info('END EXTRACTING {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)