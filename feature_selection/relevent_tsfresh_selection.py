import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.app_log import logger as app_logger
from utils.labels_extraction import known_labels_extractor
from feature_selection import test_feature_selection
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def select(dataset, clusters_number):

    app_logger.info('STARTED [RELEVANT TSFRESH Selection] on {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving relevant feature extracted by tsfresh from the pickles on the disk
    relevant_features_train = pd.read_pickle('../Pickle/RelevantFeatures/Train/{0}.pkl'.format(dataset))
    relevant_features_test = pd.read_pickle('../Pickle/RelevantFeatures/Test/{0}.pkl'.format(dataset))

    app_logger.info('Relevant features trainset shape: {0}'.format(relevant_features_train.shape), extra = LOGGER_EXTRA_OBJECT)
    app_logger.info('Relevant features testset shape: {0}'.format(relevant_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving known labels of the test set
    known_labels = known_labels_extractor.extract_known_labels('../Datasets/{0}/{0}_TEST.tsv'.format(dataset))

    # Running k-means on dataframes obtained from the pickles
    test_feature_selection.testFeatureSelectionWithKMeans('RELEVANT TSFRESH', relevant_features_train.shape[1], dataset, 
        relevant_features_train.values, relevant_features_test.values, clusters_number, known_labels)
    
    app_logger.info('ENDED [RELEVANT TSFRESH Selection] on {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)


# Testing
select('TwoPatterns', 4)