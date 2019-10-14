import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.app_log import logger as app_logger
from skfeature.utility import construct_W, sparse_learning
from skfeature.function.sparse_learning_based import MCFS
from utils.labels_extraction import known_labels_extractor
from feature_selection import test_feature_selection
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def select(dataset, features_number, clusters_number):

    app_logger.info('STARTED [MCFS Selection] on {0} with features number = {1}'.format(dataset, features_number), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving all feature extracted by tsfresh from the pickles on the disk
    all_features_train = pd.read_pickle('../Pickle/AllFeatures/Train/{0}.pkl'.format(dataset))
    all_features_test = pd.read_pickle('../Pickle/AllFeatures/Test/{0}.pkl'.format(dataset))

    app_logger.info('All features trainset shape: {0}'.format(all_features_train.shape), extra = LOGGER_EXTRA_OBJECT)
    app_logger.info('All features testset shape: {0}'.format(all_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

    # np.savetxt(r'testDataFrame.txt', all_features_test.values, fmt='%d')

    # Building matrix W for MCFS algorithm
    kwargs = {
        'metric': 'euclidean',
        'neighbor_mode': 'knn',
        'weight_mode': 'heat_kernel',
        'k': 5,
        't': 1
    }
    W = construct_W.construct_W(all_features_train.values, **kwargs)

    # MCFS gives a weight to each features
    kwargs = {
        'W': W,
        'n_clusters': clusters_number
    }
    weighted_features = MCFS.mcfs(all_features_train.values, features_number, **kwargs)

    # Ordering the features according to their weight
    ordered_features = MCFS.feature_ranking(weighted_features)

    # Getting only the first 'features_number' features
    selected_features = ordered_features[0:features_number]

    # Getting names of selected features
    names_selected_features = []
    for feature_index in selected_features:
        names_selected_features.append(all_features_train.columns[feature_index])

    # Selected only the selected features on the train set
    selected_features_train = all_features_train.loc[:, names_selected_features]
    app_logger.info('Selected features Train: {0}'.format(selected_features_train.shape), extra = LOGGER_EXTRA_OBJECT)

    # Selected only the selected features on the test set
    selected_features_test = all_features_test.loc[:, names_selected_features]
    app_logger.info('Selected features Test: {0}'.format(selected_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving known labels of the test set
    known_labels = known_labels_extractor.extract_known_labels('../Datasets/{0}/{0}_TEST.tsv'.format(dataset))

    # np.savetxt(r'selectedTestDataFrame.txt', selected_features_test.values)

    # Running k-means according to selected features
    test_feature_selection.testFeatureSelectionWithRepeatedKMeans('MCFS', features_number, dataset, 
        selected_features_train.values, selected_features_test.values, clusters_number, known_labels)

    app_logger.info('ENDED [MCFS Selection] on {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)


# Testing
select('TwoPatterns', 20, 4)