import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.app_log import logger as app_logger
from feature_selection import test_feature_selection
import pandas as pd
import numpy as np

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def select(dataset, features_number, clusters_number):
    app_logger.info('STARTED [Corr Selection] on {0} with features number = {1}'.format(dataset, features_number), extra = LOGGER_EXTRA_OBJECT)

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

    # Selecting indipendent columns and the target column of the train set
    indipendent_columns_train = all_features_train.iloc[:, 1:]
    target_column_train = all_features_train.iloc[:, 0]
    # Selecting indipendent columns and the target column of the test set
    indipendent_columns_test = all_features_test.iloc[:, 1:]
    known_labels_test = all_features_test.iloc[:, 0]

    dfcolumns = pd.DataFrame(indipendent_columns_train.columns)

    # Correlation Matrix
    # data = all_features_train
    data = all_features_train.astype(float) # Otherwise, don't consider target column beacuse its type is integer (and not float)
    corrmat = data.corr()
    dfcorr_target = pd.DataFrame(corrmat[['target']].iloc[1:].values)

    # Creating dataframe which contains columns names and correlation values
    dfscores = pd.concat([dfcolumns, dfcorr_target], axis=1)
    dfscores.columns = ['feature_name', 'target_corr']
    dfscores = dfscores.dropna(axis=0)
    
    # Converting Corr column
    dfscores[['target_corr']] = abs(dfscores[['target_corr']])

    dfscores = dfscores.sort_values(by='target_corr', ascending=False)

    top_k_scores = dfscores.head(features_number)
    app_logger.info(top_k_scores, extra = LOGGER_EXTRA_OBJECT)

    selected_features_names = top_k_scores['feature_name'].values
    selected_features_train = indipendent_columns_train.loc[:, selected_features_names]
    selected_features_test = indipendent_columns_test.loc[:, selected_features_names]


    '''
    # Pickles for rfd
    if selected_features_train.shape[0] > 1000:
        print('Test-set')
        selected_features_test.to_pickle('../rfd/Pickle_rfd/Corr/{0}.pkl'.format(dataset))
    else:
        print('Train-set')
        selected_features_train.to_pickle('../rfd/Pickle_rfd/Corr/{0}.pkl'.format(dataset))
    exit()
    '''


    # Running k-means according to selected features
    test_feature_selection.testFeatureSelectionWithRepeatedKMeans('CORRELATION', features_number, dataset, 
        selected_features_train.values, selected_features_test.values, clusters_number, known_labels_test)

    app_logger.info('ENDED [Corr Selection] on {0}'.format(dataset), extra = LOGGER_EXTRA_OBJECT)



# Testing
#select('TwoPatterns', 10, 4)