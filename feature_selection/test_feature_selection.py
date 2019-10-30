import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.result_log import logger as result_logger
from utils.evaluation.evaluation import evaluation
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def testFeatureSelectionWithRepeatedKMeans(selection_type, features_number, dataset, X_selected, X_test, n_clusters, y):

    nmi_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score, purity = evaluation(
        X_selected = X_selected, X_test = X_test, n_clusters = n_clusters, y = y)

    for i in range(0, 20):
        new_nmi_score, new_silhouette_score, new_davies_bouldin_score, new_calinski_harabasz_score, new_purity = evaluation(
            X_selected = X_selected, X_test = X_test, n_clusters = n_clusters, y = y)

        if (new_nmi_score >= nmi_score
                and new_silhouette_score >= silhouette_score 
                and new_davies_bouldin_score <= davies_bouldin_score
                and new_purity >= purity 
                and new_calinski_harabasz_score >= calinski_harabasz_score):
            nmi_score = new_nmi_score
            silhouette_score = new_silhouette_score
            davies_bouldin_score = new_davies_bouldin_score
            calinski_harabasz_score = new_calinski_harabasz_score
            purity = new_purity

    result_logger.info('[{1}] K-means scores on {0} with {2} features'.format(dataset, selection_type, features_number), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('Davies Bouldin score: {0}'.format(float(round(davies_bouldin_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('Silhouette score: {0}'.format(float(round(silhouette_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('Calinski Harabasz score: {0}'.format(float(round(calinski_harabasz_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('NMI score: {0}'.format(float(round(nmi_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('Purity: {0}'.format(float(round(purity, 5))), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('[END Result]', extra = LOGGER_EXTRA_OBJECT)

def testFeatureSelectionWithKMeans(selection_type, features_number, dataset, X_selected, X_test, n_clusters, y):

    nmi_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score, purity = evaluation(
        X_selected = X_selected, X_test = X_test, n_clusters = n_clusters, y = y)

    result_logger.info('[{1}] K-means scores on {0} with {2} features'.format(dataset, selection_type, features_number), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('Davies Bouldin score: {0}'.format(float(round(davies_bouldin_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('Silhouette score: {0}'.format(float(round(silhouette_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('Calinski Harabasz score: {0}'.format(float(round(calinski_harabasz_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('NMI score: {0}'.format(float(round(nmi_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('Purity: {0}'.format(float(round(purity, 5))), extra = LOGGER_EXTRA_OBJECT)
    result_logger.info('[END Result]', extra = LOGGER_EXTRA_OBJECT)