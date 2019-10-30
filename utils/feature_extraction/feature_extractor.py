import os
import sys
sys.path.append(os.path.abspath('..'))

from tsfresh import extract_relevant_features, extract_features, select_features
from utils.data_cleansing import drop_columns, splitting_dataset
from logs.extraction_log import logger
# from tsfresh.feature_selection.relevance import calculate_relevance_table
import pandas as pd
import utils.feature_extraction.utility_feature_extractor as util

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def extract_dataset_features(dataset):

    listOut,series = util.adaptTimeSeries(dataset)
    df = extract_features(listOut, column_id='id', column_sort='time')
    del listOut
    df = drop_columns.drop_columns_with_nan_values(df)
    df = pd.concat([pd.DataFrame(series, columns=['target']), df], axis=1)
    logger.info('All dataset shape (including target column): {0}'.format(df.shape), extra = LOGGER_EXTRA_OBJECT)

    train_set,test_set = splitting_dataset.split_dataset(df)
    logger.info('All train set shape (including target column): {0}, All test set shape (including target column): {1}'.format(train_set.shape, test_set.shape), extra = LOGGER_EXTRA_OBJECT)
    
    indipendent_columns_train = train_set.iloc[:, 1:]
    target_column_train = train_set.iloc[:, 0]
    indipendent_columns_test = test_set.iloc[:, 1:]
    target_column_test = test_set.iloc[:, 0]

    logger.info('All features train set: {0}, All features test set: {1}'.format(indipendent_columns_train.shape, indipendent_columns_test.shape), extra = LOGGER_EXTRA_OBJECT)
    logger.info('All features features (including target column) train set: {0}, All features (including target column) test set: {1}'.format(train_set.shape, test_set.shape), extra = LOGGER_EXTRA_OBJECT)
    
    train_set.to_pickle('../Pickle/AllFeatures/Train/{0}.pkl'.format(dataset))
    test_set.to_pickle('../Pickle/AllFeatures/Test/{0}.pkl'.format(dataset))
    
    extract_relevant_dataset_features(dataset, indipendent_columns_train, target_column_train, indipendent_columns_test, target_column_test)

    '''
    # Deleting columns (features) with NaN value
    all_features_train = indipendent_columns_train.dropna(axis = 1)
    all_features_test = indipendent_columns_test.dropna(axis = 1)
    logger.info('All features [WITHOUT NaN values] train set: {0}, All features [WITHOUT NaN values] test set: {1}'.format(all_features_train.shape, all_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

    extract_relevant_dataset_features(dataset, all_features_train, target_column_train, all_features_test, target_column_test)

    # Selecting common features
    all_common_features = all_features_train.columns.intersection(all_features_test.columns)

    # For each row selects only common features
    all_features_train = all_features_train.loc[:, all_common_features]
    all_features_test = all_features_test.loc[:, all_common_features]
    logger.info('All common features (with no nan) train set: {0}, All common features (with no nan) test set: {1}'.format(all_features_train.shape, all_features_test.shape), extra = LOGGER_EXTRA_OBJECT)
    all_features_train = pd.concat([pd.DataFrame(target_column_train, columns=['target']), all_features_train], axis=1)
    all_features_test = pd.concat([pd.DataFrame(target_column_test, columns=['target']), all_features_test], axis=1)
    logger.info('All common features (including target column) train set: {0}, All common features (including target column) test set: {1}'.format(all_features_train.shape, all_features_test.shape), extra = LOGGER_EXTRA_OBJECT)
    '''

def extract_relevant_dataset_features(dataset, all_features_train, target_column_train, all_features_test, target_column_test):

    relevant_features_train = select_features(all_features_train, target_column_train)
    relevant_features_test = select_features(all_features_test, target_column_test)
    logger.info('Relevant features train set: {0}, Relevant features test set: {1}'.format(relevant_features_train.shape, relevant_features_test.shape), extra = LOGGER_EXTRA_OBJECT)
    
    '''
    # Deleting columns (features) with NaN value
    relevant_features_train = relevant_features_train.dropna(axis = 1)
    relevant_features_test = relevant_features_test.dropna(axis = 1)
    logger.info('Relevant features [WITHOUT NaN values] train set: {0}, Relevant features [WITHOUT NaN values] test set: {1}'.format(relevant_features_train.shape, relevant_features_test.shape), extra = LOGGER_EXTRA_OBJECT)
    '''

    # Selecting common features
    relevant_common_features = relevant_features_train.columns.intersection(relevant_features_test.columns)

    # For each row selects only common features
    relevant_features_train = relevant_features_train.loc[:, relevant_common_features]
    relevant_features_test = relevant_features_test.loc[:, relevant_common_features]
    logger.info('Relevant common features train set: {0}, Relevant common features test set: {1}'.format(relevant_features_train.shape, relevant_features_test.shape), extra = LOGGER_EXTRA_OBJECT)
    relevant_features_train = pd.concat([pd.DataFrame(target_column_train, columns=['target']), relevant_features_train], axis=1)
    relevant_features_test = pd.concat([pd.DataFrame(target_column_test, columns=['target']), relevant_features_test], axis=1)
    logger.info('Relevant common features (including target column) train set: {0}, Relevant common features (including target column) test set: {1}'.format(relevant_features_train.shape, relevant_features_test.shape), extra = LOGGER_EXTRA_OBJECT)

    relevant_features_train.to_pickle('../Pickle/RelevantFeatures/Train/{0}.pkl'.format(dataset))
    relevant_features_test.to_pickle('../Pickle/RelevantFeatures/Test/{0}.pkl'.format(dataset))

'''
def extract_all_dataset_features(dataset, listOutTrain, listOutTest):

    all_features_train = extract_features(listOutTrain, column_id = 'id', column_sort = 'time')
    all_features_test = extract_features(listOutTest, column_id = 'id', column_sort = 'time')
    print('All features train set: {0}, All features test set: {1}'.format(all_features_train.shape, all_features_test.shape))

    # Deleting columns (features) with NaN value
    all_features_train = all_features_train.dropna(axis = 1)
    all_features_test = all_features_test.dropna(axis = 1)
    print('All features [WITHOUT NaN values] train set: {0}, All features [WITHOUT NaN values] test set: {1}'.format(all_features_train.shape, all_features_test.shape))

    # Selecting common features
    all_common_features = all_features_train.columns.intersection(all_features_test.columns)

    # For each row selects only common features
    all_features_train = all_features_train.loc[:, all_common_features]
    all_features_test = all_features_test.loc[:, all_common_features]
    print('All common features (with no nan) train set: {0}, All common features (with no nan) test set: {1}'.format(all_features_train.shape, all_features_test.shape))

    all_features_train.to_pickle('../Pickle/AllFeatures/Train/{0}.pkl'.format(dataset))
    all_features_test.to_pickle('../Pickle/AllFeatures/Test/{0}.pkl'.format(dataset))

def OLD_extract_relevant_dataset_features(dataset, listOutTrain, seriesTrain, listOutTest, seriesTest):

    relevant_features_train = extract_relevant_features(listOutTrain, seriesTrain, column_id = 'id', column_sort = 'time')
    relevant_features_test = extract_relevant_features(listOutTest, seriesTest, column_id = 'id', column_sort = 'time')
    print('Relevant features train set: {0}, Relevant features test set: {1}'.format(relevant_features_train.shape, relevant_features_test.shape))

    # Deleting columns (features) with NaN value
    relevant_features_train = relevant_features_train.dropna(axis = 1)
    relevant_features_test = relevant_features_test.dropna(axis = 1)
    print('Relevant features [WITHOUT NaN values] train set: {0}, Relevant features [WITHOUT NaN values] test set: {1}'.format(relevant_features_train.shape, relevant_features_test.shape))

    # Selecting common features
    relevant_common_features = relevant_features_train.columns.intersection(relevant_features_test.columns)

    # For each row selects only common features
    relevant_features_train = relevant_features_train.loc[:, relevant_common_features]
    relevant_features_test = relevant_features_test.loc[:, relevant_common_features]
    print('Relevant common features (with no nan) train set: {0}, Relevant common features (with no nan) test set: {1}'.format(relevant_features_train.shape, relevant_features_test.shape))

    relevant_features_train.to_pickle('../Pickle/RelevantFeatures/Train/{0}.pkl'.format(dataset))
    relevant_features_test.to_pickle('../Pickle/RelevantFeatures/Test/{0}.pkl'.format(dataset))

def NEW_extract_relevant_dataset_features(dataset, listOutTrain, seriesTrain, listOutTest, seriesTest):

    all_features_train = extract_features(listOutTrain, column_id = 'id', column_sort = 'time')
    all_features_test = extract_features(listOutTest, column_id = 'id', column_sort = 'time')
    logging.info('All features train set: {0}, All features test set: {1}'.format(all_features_train.shape, all_features_test.shape))

    relevant_features_train = select_features(all_features_train, seriesTrain)
    relevant_features_test = select_features(all_features_test, seriesTest)
    logging.info('Relevant features train set: {0}, Relevant features test set: {1}'.format(relevant_features_train.shape, relevant_features_test.shape))

    # Deleting columns (features) with NaN value
    relevant_features_train = relevant_features_train.dropna(axis = 1)
    relevant_features_test = relevant_features_test.dropna(axis = 1)
    logging.info('Relevant features [WITHOUT NaN values] train set: {0}, Relevant features [WITHOUT NaN values] test set: {1}'.format(relevant_features_train.shape, relevant_features_test.shape))

    # Selecting common features
    relevant_common_features = relevant_features_train.columns.intersection(relevant_features_test.columns)

    # For each row selects only common features
    relevant_features_train = relevant_features_train.loc[:, relevant_common_features]
    relevant_features_test = relevant_features_test.loc[:, relevant_common_features]
    logging.info('Relevant common features (with no nan) train set: {0}, Relevant common features (with no nan) test set: {1}'.format(relevant_features_train.shape, relevant_features_test.shape))
'''

'''
extract_dataset_features('FordA')


def extract_relevant_dataset_features(dataset):


if __name__ == '__main__': 

	# Series ti restituisce anche le classi di appartenenza perché vi servono se volete estrarre
	# le features rilevanti
    listOut,series = util.adaptTimeSeries("C:/Users/Donato/Desktop/UCRArchive_2018/ECG5000/ECG5000_TEST.tsv")
    
    # Questa è la funzione che vi estrae quelle interessanti
    features_filtered_direct = extract_relevant_features(listOut,series, column_id='id', column_sort='time')

    # Questa è la funzione che vi estrae tutte le features
    features_filtered_direct = extract_features(listOut,column_id='id', column_sort='time')
    print(len(features_filtered_direct))
    
    # Questa funzione consente di salvare le features che avete estratto, senza doverle riestrarre di nuovo
    # Occhio all'estensione perché se no jastemmate pur francese, esperienza personale.
    features_filtered_direct.to_pickle("./nomeAPiacere.pkl")

    # Questa funzione invece vi consente di estrarre le features dal pickle creato
    features_filtered_direct = pd.read_pickle("./nomeAPiacere.pkl")
'''