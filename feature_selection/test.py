import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.result_log import logger
from utils.labels_extraction import known_labels_extractor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
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

        if (new_nmi_score >= nmi_score
        and new_silhouette_score >= silhouette_score 
        and new_davies_bouldin_score <= davies_bouldin_score 
        and new_calinski_harabasz_score >= calinski_harabasz_score 
        and new_purity >= purity):
            nmi_score = new_nmi_score
            silhouette_score = new_silhouette_score
            davies_bouldin_score = new_davies_bouldin_score
            calinski_harabasz_score = new_calinski_harabasz_score
            purity = new_purity

    logger.info('NMI score: {0}'.format(float(round(nmi_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Silhouette score: {0}'.format(float(round(silhouette_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Davies Bouldin score: {0}'.format(float(round(davies_bouldin_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Calinski Harabasz score: {0}'.format(float(round(calinski_harabasz_score, 5))), extra = LOGGER_EXTRA_OBJECT)
    logger.info('Purity: {0}'.format(float(round(purity, 5))), extra = LOGGER_EXTRA_OBJECT)
    logger.info('END [MCFS Features Selection] {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)

def test(dataset, features_number, clusters_number):
    # Retrieving all feature extracted by tsfresh from the pickles on the disk
    all_features_train = pd.read_pickle('../Pickle/AllFeatures/Train/{0}.pkl'.format(dataset))
    all_features_test = pd.read_pickle('../Pickle/AllFeatures/Test/{0}.pkl'.format(dataset))

    # Selecting indipendent columns and the target column of the set
    indipendent_columns = all_features_train
    target_column = known_labels_extractor.extract_known_labels('../Datasets/{0}/{0}_TRAIN.tsv'.format(dataset))

    dfcolumns = pd.DataFrame(indipendent_columns.columns)

    # Feature variance
    features_variance = indipendent_columns.var()
    dfvariance = pd.DataFrame(features_variance.values)

    # Feature Importance
    model = ExtraTreesClassifier()
    model.fit(indipendent_columns, target_column)
    dfimportance = pd.DataFrame(model.feature_importances_)

    # Correlation Matrix
    data = pd.concat([pd.DataFrame(target_column, columns = ['Target']), indipendent_columns], axis = 1)
    corrmat = data.corr()
    dfcorr_target = pd.DataFrame(corrmat[['Target']].iloc[1:].values)

    # Creating dataframe which contains the three scores
    dfscores = pd.concat([dfcolumns, dfvariance, dfimportance, dfcorr_target], axis=1)
    dfscores.columns = ['feature_name', 'variance', 'importance', 'target_corr']
    dfscores = dfscores.dropna(axis = 0)
    
    # Reducing all columns in [0, 1] range
    new_max = 1
    new_min = 0
    new_range = new_max - new_min
    # Converting Variance column
    v_old_max = dfscores[['variance']].max()
    v_old_min = dfscores[['variance']].min()
    v_old_range = v_old_max - v_old_min
    dfscores[['variance']] = (((dfscores[['variance']] - v_old_min) * new_range) / v_old_range) + new_min
    # Importance column is already in [0, 1] range
    # Converting Corr column
    dfscores[['target_corr']] = abs(dfscores[['target_corr']])

    # Calculating the average of the three scores
    # dfscores['average'] = dfscores[['variance', 'importance', 'target_corr']].mean(axis=1)
    dfscores['weighted_average'] = 0.20 * dfscores['variance'] + 0.10 * dfscores['importance'] + 0.70 * dfscores['target_corr']
    dfscores = dfscores.sort_values(by='weighted_average', ascending=False)
    print(dfscores)

    top_k_scores = dfscores.head(features_number)
    print(top_k_scores)


    selected_features_names = top_k_scores['feature_name'].values
    selected_features_train = all_features_train.loc[:, selected_features_names]
    # print(selected_features_train.columns)
    selected_features_test = all_features_test.loc[:, selected_features_names]

    # Retrieving known labels of the test set
    known_labels = known_labels_extractor.extract_known_labels('../Datasets/{0}/{0}_TEST.tsv'.format(dataset))

    # Running k-means according to selected features
    testFeatureSelection(dataset, selected_features_train.values, selected_features_test.values, clusters_number, known_labels)


test("TwoPatterns", 10, 4)