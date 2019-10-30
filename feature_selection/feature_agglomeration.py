import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.app_log import logger as app_logger
from sklearn import cluster
from feature_selection import test_feature_selection
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def agglomerate(dataset, features_number, clusters_number):
    app_logger.info('STARTED [Feature Agglomeration] on {0} with features number = {1}'.format(dataset, features_number), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving all feature extracted by tsfresh from the pickles on the disk
    current_dir = os.getcwd().split('\\')[-1]
    projet_dir = 'MCFS-Unsupervisioned-Feature-Selection'
    if current_dir == projet_dir:
        all_features_train = pd.read_pickle('Pickle/AllFeatures/Train/{0}.pkl'.format(dataset))
        all_features_test = pd.read_pickle('Pickle/AllFeatures/Test/{0}.pkl'.format(dataset))
    else:
        all_features_train = pd.read_pickle('../Pickle/AllFeatures/Train/{0}.pkl'.format(dataset))
        all_features_test = pd.read_pickle('../Pickle/AllFeatures/Test/{0}.pkl'.format(dataset))

    app_logger.info('All features (including target column) trainset shape: {0}'.format(all_features_train.shape), extra = LOGGER_EXTRA_OBJECT)
    app_logger.info('All features (including target column) testset shape: {0}'.format(all_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving indipendent columns of both set and known labels of the test set
    indipendent_columns_train = all_features_train.iloc[:, 1:]
    indipendent_columns_test = all_features_test.iloc[:, 1:]
    known_labels_test = all_features_test.iloc[:, 0]

    agglomeration = cluster.FeatureAgglomeration(n_clusters=features_number)
    agglomeration.fit(indipendent_columns_train)
    reduced_train = agglomeration.transform(indipendent_columns_train)
    reduced_test = agglomeration.transform(indipendent_columns_test)
    app_logger.info('Reduced train set: {0}'.format(reduced_train), extra = LOGGER_EXTRA_OBJECT)
    app_logger.info('Reduced test set: {0}'.format(reduced_test), extra = LOGGER_EXTRA_OBJECT)

    # Running k-means according to selected features
    test_feature_selection.testFeatureSelectionWithRepeatedKMeans('AGGLOMERATION', features_number, dataset, 
        reduced_train, reduced_test, clusters_number, known_labels_test)

    app_logger.info('ENDED [Feature Agglomeration] on {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)

# Testing
# agglomerate('TwoPatterns', 20, 4)

