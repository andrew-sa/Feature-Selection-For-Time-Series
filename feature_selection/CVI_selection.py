import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.app_log import logger as app_logger
from utils.labels_extraction import known_labels_extractor
from sklearn.ensemble import ExtraTreesClassifier
from feature_selection import test_feature_selection
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def select(dataset, features_number, clusters_number):
    app_logger.info('STARTED [CVI Selection] on {0} with features number = {1}'.format(dataset, features_number), extra = LOGGER_EXTRA_OBJECT)

    # Retrieving all feature extracted by tsfresh from the pickles on the disk
    all_features_train = pd.read_pickle('../Pickle/AllFeatures/Train/{0}.pkl'.format(dataset))
    all_features_test = pd.read_pickle('../Pickle/AllFeatures/Test/{0}.pkl'.format(dataset))

    app_logger.info('All features trainset shape: {0}'.format(all_features_train.shape), extra = LOGGER_EXTRA_OBJECT)
    app_logger.info('All features testset shape: {0}'.format(all_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

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

    # Calculating the weighted average of the three scores
    # dfscores['average'] = dfscores[['variance', 'importance', 'target_corr']].mean(axis=1)
    dfscores['weighted_average'] = 0.20 * dfscores['variance'] + 0.10 * dfscores['importance'] + 0.70 * dfscores['target_corr']
    dfscores = dfscores.sort_values(by='weighted_average', ascending=False)

    top_k_scores = dfscores.head(features_number)
    app_logger.info(top_k_scores, extra = LOGGER_EXTRA_OBJECT)

    selected_features_names = top_k_scores['feature_name'].values
    selected_features_train = all_features_train.loc[:, selected_features_names]
    selected_features_test = all_features_test.loc[:, selected_features_names]

    # Retrieving known labels of the test set
    known_labels = known_labels_extractor.extract_known_labels('../Datasets/{0}/{0}_TEST.tsv'.format(dataset))

    # Running k-means according to selected features
    test_feature_selection.testFeatureSelectionWithRepeatedKMeans('CVI', features_number, dataset, 
        selected_features_train.values, selected_features_test.values, clusters_number, known_labels)

    app_logger.info('ENDED [CVI Selection] on {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)



# Testing
select("TwoPatterns", 20, 4)