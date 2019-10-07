import os
import sys

from feature_selection import all_tsfresh_features, relevent_tsfresh_features, MCFS_features

from logs.app_log import logger

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
    # logging.basicConfig(filename = './Logs/myapp.log', format = '%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s', level = logging.DEBUG)
    
    # logging.info('we')

    # logger.info('I\'m the main', extra = LOGGER_EXTRA_OBJECT)
    
    dataset = sys.argv[1]
    feature_number = int(sys.argv[2])
    clusters_number = CLUSTERS_NUMBERS[dataset]

    all_tsfresh_features.select(dataset, clusters_number)
    relevent_tsfresh_features.select(dataset, clusters_number)
    MCFS_features.selectFeatures(dataset, features_number, clusters_number)


if __name__ == '__main__':
    main()