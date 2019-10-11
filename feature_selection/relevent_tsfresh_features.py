import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.result_log import logger
from utils.labels_extraction import known_labels_extractor
from utils.evaluation import evaluation
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def select(dataset, clusters_number):

    logger.info('START [RELEVANT TSFRESH Features] {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving relevant feature extracted by tsfresh from the pickles on the disk
    relevant_features_train = pd.read_pickle('../Pickle/RelevantFeatures/Train/{0}.pkl'.format(dataset))
    relevant_features_test = pd.read_pickle('../Pickle/RelevantFeatures/Test/{0}.pkl'.format(dataset))

    logger.info('Relevant features trainset shape: {0}'.format(relevant_features_train.shape), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Relevant features testset shape: {0}'.format(relevant_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving known labels of the test set
    known_labels = known_labels_extractor.extract_known_labels('../Datasets/{0}/{0}_TEST.tsv'.format(dataset))

    # Running k-means on dataframes obtained from the pickles
    nmi_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score, purity = evaluation.evaluation(
        X_selected = relevant_features_train.values, X_test = relevant_features_test.values, n_clusters = clusters_number, y = known_labels)
    
    logger.info('NMI score: {0}'.format(float(round(nmi_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Silhouette score: {0}'.format(float(round(silhouette_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Davies Bouldin score: {0}'.format(float(round(davies_bouldin_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Calinski Harabasz score: {0}'.format(float(round(calinski_harabasz_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Purity: {0}'.format(float(round(purity, 5))), extra = LOGGER_EXTRA_OBJECT)
    logger.info('END [RELEVANT TSFRESH Features] {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)

# Testing
select('TwoPatterns', 4)