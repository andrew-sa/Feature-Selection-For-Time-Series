import os
import sys

from feature_selection import all_tsfresh_selection, relevent_tsfresh_selection, MCFS_selection, CVI_selection

from logs.app_log import logger as app_logger

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

CLUSTERS_NUMBERS = {
    'ChlorineConcentration': 3,
    'ECG200': 2,
    'ECG5000': 5,
    'FordA': 2,
    'FordB': 2,
    'PhalangesOutlinesCorrect': 3,
    'RefrigerationDevices': 3,
    'TwoLeadECG': 2,
    'TwoPatterns': 4 
}

def main(): 
    dataset = sys.argv[1]
    feature_number = int(sys.argv[2])
    clusters_number = CLUSTERS_NUMBERS[dataset]

    app_logger.info('STARTED {0} with {1} features'.format(dataset, feature_number), extra = LOGGER_EXTRA_OBJECT)
    all_tsfresh_features.select(dataset, clusters_number)
    relevent_tsfresh_features.select(dataset, clusters_number)
    MCFS_features.select(dataset, features_number, clusters_number)
    CVI_features.select(dataset, features_number, clusters_number)
    app_logger.info('ENDED {0} with {1} features'.format(dataset, feature_number), extra = LOGGER_EXTRA_OBJECT)

if __name__ == '__main__':
    main()