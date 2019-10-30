import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.app_log import logger as app_logger
import numpy as np
# import sklearn.utils.linear_assignment_ as la
# from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score, contingency_matrix
from sklearn.cluster import KMeans

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

'''
def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            G[i, j] = np.count_nonzero(ss & tt)

    A = la.linear_assignment(-G)

    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        new_l2[l2 == label2[A[i][1]]] = label1[A[i][0]]
    return new_l2.astype(int)
'''

def purity(known_labels, labels_predict):
    contingency_table = contingency_matrix(known_labels, labels_predict)
    purity = (np.sum(np.amax(contingency_table, axis = 0)) / np.sum(contingency_table))
    return purity

def evaluation(X_selected, X_test, n_clusters, y):
    """
    This function calculates ARI, ACC and NMI of clustering results

    Input
    -----
    X_selected: {numpy array}, shape (n_samples, n_selected_features}
            input data on the selected features
    n_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels

    Output
    ------
    nmi: {float}
        Normalized Mutual Information
    acc: {float}
        Accuracy
    """
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                     tol=0.0001, precompute_distances=True, verbose=0,
                     random_state=None, copy_x=True, n_jobs=1)

    k_means.fit(X_selected)
    y_predict = k_means.predict(X_test)
    
    # calculate NMI
    nmi = normalized_mutual_info_score(y, y_predict, average_method='arithmetic')

    # calculate Silhouette score
    try:
        sil = silhouette_score(X_test, y_predict, metric='euclidean')
    except ValueError:
        sil = float('nan')
        app_logger.warning('K-means lables are {0}; but y_predict are: {1}. Silhouette score requires predicts in 2 or more clusters.'.format(np.unique(k_means.labels_), np.unique(y_predict)), extra = LOGGER_EXTRA_OBJECT)

    # calculate Davies Bouldin 
    try:
        db = davies_bouldin_score(X_test, y_predict)
    except ValueError:
        db = float('nan')
        app_logger.warning('K-means lables are {0}; but y_predict are: {1}. Davies Bouldin score requires predicts in 2 or more clusters.'.format(np.unique(k_means.labels_), np.unique(y_predict)), extra = LOGGER_EXTRA_OBJECT)

    # calculate Calinski Harabasz score
    try:
        ch = calinski_harabasz_score(X_test, y_predict)
    except ValueError:
        ch = float('nan')
        app_logger.warning('K-means lables are {0}; but y_predict are: {1}. Calinski Harabasz score requires predicts in 2 or more clusters.'.format(np.unique(k_means.labels_), np.unique(y_predict)), extra = LOGGER_EXTRA_OBJECT)

    # calculate Purity
    pur = purity(y, y_predict)

    return nmi, sil, db, ch, pur

    '''
    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)

    return nmi, acc
    '''