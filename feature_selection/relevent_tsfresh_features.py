import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.result_log import logger
from utils.labels_extraction import known_labels_extractor
from utils.evaluation import evaluation
import pandas as pd
import numpy as np

def select(dataset, clusters_number):

    logger.info('START [RELEVANT TSFRESH Features] {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving relevant feature extracted by tsfresh from the pickles on the disk
    relevant_features_train = pd.read_pickle('../Pickle/RelevantFeatures/Train/{0}.pkl'.format(dataset))
    relevant_features_test = pd.read_pickle('../Pickle/RelevantFeatures/Test/{0}.pkl'.format(dataset))

    logger.info('Features number: {0}'.format(relevant_features_test.shape[1]), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving known labels of the test set
    known_labels = known_labels_extractor.extract_known_labels('../Datasets/{0}/{0}_TEST.tsv'.format(dataset))

    # Running k-means on dataframes obtained from the pickles
    nmi_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score, purity = evaluation.evaluation(
        X_selected = all_features_train.values, X_test = all_features_test.values, n_clusters = clusters_number, y = known_labels)
    
    logger.info('NMI score: {0}'.format(nmi_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Silhouette score: {0}'.format(silhouette_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Davies Bouldin score: {0}'.format(davies_bouldin_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Calinski Harabasz score: {0}'.format(calinski_harabasz_score), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Purity: {0}'.format(purity), extra = LOGGER_EXTRA_OBJECT)
    logger.info('END [RELEVANT TSFRESH Features] {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)