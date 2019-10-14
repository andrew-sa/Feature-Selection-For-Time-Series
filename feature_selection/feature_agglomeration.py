import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.app_log import logger as app_logger
from sklearn import cluster
from utils.labels_extraction import known_labels_extractor
from feature_selection import test_feature_selection
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def agglomerate(dataset, features_number, clusters_number):
    app_logger.info('STARTED [Feature Agglomeration] on {0} with features number = {1}'.format(dataset, features_number), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving all feature extracted by tsfresh from the pickles on the disk
    all_features_train = pd.read_pickle('../Pickle/AllFeatures/Train/{0}.pkl'.format(dataset))
    all_features_test = pd.read_pickle('../Pickle/AllFeatures/Test/{0}.pkl'.format(dataset))

    app_logger.info('All features trainset shape: {0}'.format(all_features_train.shape), extra = LOGGER_EXTRA_OBJECT)
    app_logger.info('All features testset shape: {0}'.format(all_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

    agglomeration = cluster.FeatureAgglomeration(n_clusters=features_number)
    agglomeration.fit(all_features_train)
    reduced_train = agglomeration.transform(all_features_train)
    reduced_test = agglomeration.transform(all_features_test)
    print('Train: {0}, Test: {1}'.format(reduced_train.shape, reduced_test[[1]]))

    # Retrieving known labels of the test set
    known_labels = known_labels_extractor.extract_known_labels('../Datasets/{0}/{0}_TEST.tsv'.format(dataset))

    # Running k-means according to selected features
    test_feature_selection.testFeatureSelectionWithRepeatedKMeans('AGGLOMERATION', features_number, dataset, 
        reduced_train, reduced_test, clusters_number, known_labels)

    app_logger.info('ENDED [Feature Agglomeration] on {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)

# Testing
agglomerate('TwoPatterns', 20, 4)

