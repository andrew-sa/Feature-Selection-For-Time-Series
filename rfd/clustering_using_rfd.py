import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.kmeans_rfd_log import logger as kmeans_rfd_logger
from skfeature.utility import construct_W, sparse_learning
from skfeature.function.sparse_learning_based import MCFS
from utils.evaluation.evaluation import evaluation
import pandas as pd
import numpy as np

CLUSTERS_NUMBERS = {
    'ChlorineConcentration': 3,
    'ECG200': 2,
    'ECG5000': 5,
    'FordA': 2,
    'FordB': 2,
    'PhalangesOutlinesCorrect': 3,
    'RefrigerationDevices': 3,
    'TwoLeadECG': 2,
    'TwoPatterns': 4 
}

def run_kmeans(features_number, X_selected, X_test, n_clusters, y):
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

    kmeans_rfd_logger.info('(Features number: {0})'.format(features_number))
    kmeans_rfd_logger.info('Davies Bouldin score: {0}'.format(float(round(davies_bouldin_score, 5))))
    kmeans_rfd_logger.info('Silhouette score: {0}'.format(float(round(silhouette_score, 5))))
    kmeans_rfd_logger.info('Calinski Harabasz score: {0}'.format(float(round(calinski_harabasz_score, 5))))
    kmeans_rfd_logger.info('NMI score: {0}'.format(float(round(nmi_score, 5))))
    kmeans_rfd_logger.info('Purity: {0}'.format(float(round(purity, 5))))

def mcfs(train_set, test_set, features_number, clusters_number):
    # Features to delete
    features_to_delete = []
    for i in range(3, len(sys.argv)):
        features_to_delete.append(sys.argv[i])

    # Retrieving indipendent columns of both set and known labels of the test set
    indipendent_columns_train = train_set.iloc[:, 1:]
    indipendent_columns_test = test_set.iloc[:, 1:]
    known_labels_test = test_set.iloc[:, 0]

    # Building matrix W for MCFS algorithm
    kwargs = {
        'metric': 'euclidean',
        'neighbor_mode': 'knn',
        'weight_mode': 'binary',
        'k': 3
    }
    W = construct_W.construct_W(indipendent_columns_train.values, **kwargs)

    # MCFS gives a weight to each features
    kwargs = {
        'W': W,
        'n_clusters': clusters_number
    }
    weighted_features = MCFS.mcfs(indipendent_columns_train.values, features_number, **kwargs)

    # Ordering the features according to their weight
    ordered_features = MCFS.feature_ranking(weighted_features)

    # Getting only the first 'features_number' features
    selected_features = ordered_features[0:features_number]

    # Getting names of selected features
    names_selected_features = []
    for feature_index in selected_features:
        names_selected_features.append(indipendent_columns_train.columns[feature_index])
    
    # Deleting "feature to delete"
    names_selected_features = [feature for feature in names_selected_features if feature not in features_to_delete]

    if len(names_selected_features) != len(selected_features) - len(features_to_delete):
        kmeans_rfd_logger.error('One or more feature "to delete" is/are not correct.')
    else:
        # Selected only the selected features on the train set
        selected_features_train = indipendent_columns_train.loc[:, names_selected_features]

        # Selected only the selected features on the test set
        selected_features_test = indipendent_columns_test.loc[:, names_selected_features]

        kmeans_rfd_logger.info('(Deleted features: {0})'.format(features_to_delete))

        # Running k-means according to selected features
        run_kmeans(len(names_selected_features), selected_features_train.values, 
            selected_features_test.values, clusters_number, known_labels_test)



def correlation(train_set, test_set, features_number, clusters_number):
    # Features to delete
    features_to_delete = []
    for i in range(3, len(sys.argv)):
        features_to_delete.append(sys.argv[i])

    # Selecting indipendent columns and the target column of the train set
    indipendent_columns_train = train_set.iloc[:, 1:]
    target_column_train = train_set.iloc[:, 0]
    # Selecting indipendent columns and the target column of the test set
    indipendent_columns_test = test_set.iloc[:, 1:]
    known_labels_test = test_set.iloc[:, 0]

    dfcolumns = pd.DataFrame(indipendent_columns_train.columns)

    # Correlation Matrix
    # data = all_features_train
    data = train_set.astype(float) # Otherwise, don't consider target column beacuse its type is integer (and not float)
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
    tok_k_scores_names = top_k_scores['feature_name'].values
    
    # Deleting feature "to delate"
    selected_features_names = [feature for feature in tok_k_scores_names if feature not in features_to_delete]

    if len(selected_features_names) != len(tok_k_scores_names) - len(features_to_delete):
        kmeans_rfd_logger.error('One or more feature "to delete" is/are not correct.')
    else:
        # Selected only the selected features on the train set
        selected_features_train = indipendent_columns_train.loc[:, selected_features_names]

        # Selected only the selected features on the test set
        selected_features_test = indipendent_columns_test.loc[:, selected_features_names]

        kmeans_rfd_logger.info('(Deleted features: {0})'.format(features_to_delete))

        # Running k-means according to selected features
        run_kmeans(len(selected_features_names), selected_features_train.values, 
            selected_features_test.values, clusters_number, known_labels_test)



if __name__ == '__main__':
    if len(sys.argv) < 4:
        kmeans_rfd_logger.error('You have to insert: feature selection mode (mcfs or corr), dataset name, one or more names of the features to delete.') 
    else:
        dataset = sys.argv[2]
        if dataset in CLUSTERS_NUMBERS.keys():
            clusters_number = CLUSTERS_NUMBERS[dataset]
            train_set = pd.read_pickle('../Pickle/AllFeatures/Train/{0}.pkl'.format(dataset))
            test_set = pd.read_pickle('../Pickle/AllFeatures/Test/{0}.pkl'.format(dataset))
            if sys.argv[1].lower() == 'mcfs':
                kmeans_rfd_logger.info('[STARTED] [{0}] [MCFS]'.format(dataset))
                mcfs(train_set, test_set, 10, clusters_number)
                kmeans_rfd_logger.info('[ENDED] [{0}] [MCFS]'.format(dataset))
            elif sys.argv[1].lower() == 'corr':
                kmeans_rfd_logger.info('[STARTED] [{0}] [Corr]'.format(dataset))
                correlation(train_set, test_set, 10, clusters_number)
                kmeans_rfd_logger.info('[ENDED] [{0}] [Corr]'.format(dataset))
            else:
                kmeans_rfd_logger.error('Feature selection mode inserted is not valid.')
        else:
            kmeans_rfd_logger.error('Dataset name is not valid.')
