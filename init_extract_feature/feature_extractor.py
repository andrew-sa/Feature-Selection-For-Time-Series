import os
import sys
sys.path.append(os.path.abspath('..'))

from log_files.log import logger
from feature_extraction import estrattoreFeature
import pandas as pd
import time

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

if __name__ == '__main__':

    datasets = ['ECG200', 'ECG5000', 'FordA', 'FordB', 'ChlorineConcentration', 'PhalangesOutlinesCorrect',
                'RefrigerationDevices', 'TwoLeadECG', 'TwoPatterns']

    # for dataset in datasets:
    #     logger.info('START EXTRACTING {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)
    #     estrattoreFeature.extract_dataset_features(dataset)
    #     logger.info('END EXTRACTING {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)
    #     time.sleep(120)

    dataset = datasets[8]
    logger.info('START EXTRACTING {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)
    estrattoreFeature.extract_dataset_features(dataset)
    logger.info('END EXTRACTING {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)