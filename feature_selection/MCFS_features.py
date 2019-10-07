import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.result_log import logger
from skfeature.utility import construct_W, sparse_learning
from skfeature.function.sparse_learning_based import MCFS
from utils.labels_extraction import known_labels_extractor
from utils.evaluation.evaluation import evaluation
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def testFeatureSelection(dataset, X_selected, X_test, n_clusters, y):

    nmi_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score, purity = evaluation(
        X_selected = X_selected, X_test = X_test, n_clusters = n_clusters, y = y)

    for i in range(0, 20):
        new_nmi_score, new_silhouette_score, new_davies_bouldin_score, new_calinski_harabasz_score, new_purity = evaluation(
            X_selected = X_selected, X_test = X_test, n_clusters = n_clusters, y = y)

        if (new_nmi_score >= nmi_score and new_silhouette_score >= silhouette_score 
        and new_davies_bouldin_score <= davies_bouldin_score and new_calinski_harabasz_score >= calinski_harabasz_score 
        and new_purity >= purity):
            nmi_score = new_nmi_score
            silhouette_score = new_silhouette_score
            davies_bouldin_score = new_davies_bouldin_score
            calinski_harabasz_score = new_calinski_harabasz_score
            purity = new_purity

    logger.info('NMI score: {0}'.format(nmi_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Silhouette score: {0}'.format(silhouette_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Davies Bouldin score: {0}'.format(davies_bouldin_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Calinski Harabasz score: {0}'.format(calinski_harabasz_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Purity: {0}'.format(purity), extra = LOGGER_EXTRA_OBJECT)
    logger.info('END [MCFS Features Selection] {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)


def selectFeatures(dataset, features_number, clusters_number):

    logger.info('START [MCFS Features Selection] {0} with features number = {1}'.format(dataset, features_number), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving all feature extracted by tsfresh from the pickles on the disk
    all_features_train = pd.read_pickle('../Pickle/AllFeatures/Train/{0}.pkl'.format(dataset))
    all_features_test = pd.read_pickle('../Pickle/AllFeatures/Test/{0}.pkl'.format(dataset))

    logger.info('All train features number: {0}'.format(all_features_train.shape), extra = LOGGER_EXTRA_OBJECT)
    logger.info('All test features number: {0}'.format(all_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

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
    logger.info('Selected features Train: {0}'.format(selected_features_train.shape), extra = LOGGER_EXTRA_OBJECT)

    # Selected only the selected features on the test set
    selected_features_test = all_features_test.loc[:, names_selected_features]
    logger.info('Selected features Test: {0}'.format(selected_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving known labels of the test set
    known_labels = known_labels_extractor.extract_known_labels('../Datasets/{0}/{0}_TEST.tsv'.format(dataset))

    # np.savetxt(r'selectedTestDataFrame.txt', selected_features_test.values)

    # Running k-means according to selected features
    testFeatureSelection(dataset, selected_features_train.values, selected_features_test.values, clusters_number, known_labels)

    '''
    features_occurrence = {}

    # Excecuting MCFS several time to really get the best features 
    for i in range(0, 10):
        # MCFS gives a weight to each features
        kwargs = {
            'W': W,
            'n_clusters': clusters_number
        }
        weighted_features = MCFS.mcfs(all_features_train, features_number, **kwargs)

        # Ordering the features according to their weight
        ordered_features = sparse_learning.feature_ranking(weighted_features)
        print(ordered_features)
        print(MCFS.feature_ranking(weighted_features))
        
        # Getting only the first 'features_number' features
        ordered_features = ordered_features[0:features_number]
        print(ordered_features)

        # Updating occurrence count of the features selected by MCFS
        for feature in ordered_features:
            if feature in features_occurrence:
                features_occurrence[feature] += 1
            else:
                features_occurrence[feature] = 1
    
    print(features_occurrence)
    # ordered_features_occurrence = sorted()
    '''

'''
    # Retrieving known labels of the test set
    known_labels = known_labels_extractor.extract_known_labels('../Datasets/{0}/{0}_TEST.tsv'.format(dataset))

    # Running k-means on dataframes obtained from the pickles
    nmi_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score, purity = evaluation(
        X_selected = all_features_train.values, X_test = all_features_test.values, n_clusters = clusters_number, y = known_labels)

    logger.info('NMI score: {0}'.format(nmi_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Silhouette score: {0}'.format(silhouette_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Davies Bouldin score: {0}'.format(davies_bouldin_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Calinski Harabasz score: {0}'.format(calinski_harabasz_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Purity: {0}'.format(purity), extra = LOGGER_EXTRA_OBJECT)
    logger.info('END [ALL TSFRESH Features] {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)
'''

selectFeatures('FordB', 10, 2)