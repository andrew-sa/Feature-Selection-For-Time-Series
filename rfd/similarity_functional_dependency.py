import os
import sys
sys.path.append(os.path.abspath('..'))

from logs.rfd_log import logger as rfd_logger
from logs.discovered_rfd_log import logger as discovered_rfd_logger
from rfd import list_operations
# from sklearn import preprocessing
import pandas as pd
import numpy as np
import scipy.special as scipy_special

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

# def mean_absolute_deviation(data):
#     mean = np.mean(data)
#     absolute_deviation = 0
#     for x in data:
#         absolute_deviation += abs(x - mean)
#     mean_absolute_deviation = absolute_deviation / len(data)
#     print('MAD: {0}'.format(mean_absolute_deviation))
#     return mean_absolute_deviation

# def calculate_epsilon(data):
#     standardized_data = preprocessing.scale(data)
#     #normalized_data = preprocessing.normalize(data)
#     print('STD: {0}, {1}'.format(np.mean(standardized_data), np.var(standardized_data)))
#     #print('NORMAL: {0}, {1}'.format(np.mean(normalized_data), np.var(normalized_data)))
#     return 0


def calculate_epsilon(data):

    ''' Standard deviation method '''
    # epsilon = np.std(data)

    ''' Distance method '''
    total_distance = 0
    # count = 0
    for i in range(0, len(data)):
        for j in range(i+1, len(data)):
            total_distance += abs(data[i] - data[j])
            # count += 1
    epsilon = total_distance / (scipy_special.binom(len(data), 2))
    
    return epsilon


def discovery_one_to_one(neighbourhoods, feature_names):
    for lhs in range(0, len(neighbourhoods)):
        lhs_neighbourhoods = neighbourhoods[lhs]
        if (len(lhs_neighbourhoods) == 0):
            rfd_logger.info('Feature {0} haven\'t neighbourhoods'.format(lhs))
        else:
            for rhs in range(0, len(neighbourhoods)):
                if rhs != lhs:
                    rfd_logger.info('[TRYING] {0} --> {1}'.format(lhs, rhs))
                    rhs_neighbourhoods = neighbourhoods[rhs]
                    if len(rhs_neighbourhoods) > 0:
                        flag = False
                        discovered = True
                        for ts in range(0, len(lhs_neighbourhoods)):
                            ts_neighbourhood_of_lhs = lhs_neighbourhoods[ts]
                            ts_neighbourhood_of_rhs = rhs_neighbourhoods[ts]
                            if (len(ts_neighbourhood_of_lhs) > 0 and len(ts_neighbourhood_of_rhs) > 0):
                                flag = True
                                # ts_neighbourhood_of_lhs.sort()
                                # ts_neighbourhood_of_rhs.sort()
                                # equals = np.array_equal(ts_neighbourhood_of_lhs, ts_neighbourhood_of_rhs)
                                result = set(ts_neighbourhood_of_rhs).issuperset(set(ts_neighbourhood_of_lhs))
                                if result == False:
                                    discovered = False
                        if flag == True and discovered == True:
                            rfd_logger.info('[DISCOVERED] {0} --> {1}'.format(lhs, rhs))
                            discovered_rfd_logger.info('[DISCOVERED] {0} --> {1}'.format(lhs, rhs))
                            discovered_rfd_logger.info('{0} --> {1}'.format(feature_names[lhs], feature_names[rhs]))


            '''
            for ts in range(0, len(feature_neighbourhood)):
                ts_neighbourhood_of_lhs = feature_neighbourhood[ts]
                if (len(ts_neighbourhood_of_lhs) > 0):
                    for rhs in range((lhs+1), len(neighbourhoods)):
                        ts_neighbourhood_of_rhs = neighbourhoods[rhs][ts]
                        flag = np.array_equal(ts_neighbourhood_of_lhs, ts_neighbourhood_of_rhs)
                        if flag == True:
                            print('{0} --> {1}'.format(lhs, rhs))
            '''


def discovery_two_to_one(neighbourhoods, feature_names):
    for lhs1 in range(0, len(neighbourhoods)):
        lhs1_neighbourhoods = neighbourhoods[lhs1]
        if (len(lhs1_neighbourhoods) == 0):
            rfd_logger.info('Feature {0} haven\'t neighbourhoods'.format(lhs1))
        else: 
            for lhs2 in range((lhs1+1), len(neighbourhoods)):
            # for lhs2 in range(lhs1, len(neighbourhoods)):   #Testing i,i-->i 
                for rhs in range(0, len(neighbourhoods)):
                    if rhs != lhs1 and rhs != lhs2:
                    # if rhs == lhs1 and rhs == lhs2: #Testing i,i-->i
                        rfd_logger.info('[TRYING] {0}, {1} --> {2}'.format(lhs1, lhs2, rhs))
                        lhs2_neighbourhoods = neighbourhoods[lhs2]
                        rhs_neighbourhoods = neighbourhoods[rhs]
                        if len(lhs2_neighbourhoods) > 0 and len(rhs_neighbourhoods) > 0:
                            flag = False
                            discovered = True
                            for ts in range(0, len(lhs1_neighbourhoods)):
                                # ts_neighbourhood_of_lhs = list_operations.intersection_between_two(lhs1_neighbourhoods[ts], lhs2_neighbourhoods[ts])
                                # ts_neighbourhood_of_rhs = rhs_neighbourhoods[ts]
                                ts_neighbourhood_of_lhs = set(lhs1_neighbourhoods[ts]).intersection(set(lhs2_neighbourhoods[ts]))
                                ts_neighbourhood_of_rhs = set(rhs_neighbourhoods[ts])
                                if (len(ts_neighbourhood_of_lhs) > 0 and len(ts_neighbourhood_of_rhs) > 0):
                                    flag = True
                                    # ts_neighbourhood_of_lhs.sort()
                                    # ts_neighbourhood_of_rhs.sort()
                                    # equals = np.array_equal(ts_neighbourhood_of_lhs, ts_neighbourhood_of_rhs)
                                    result = ts_neighbourhood_of_rhs.issuperset(ts_neighbourhood_of_lhs)
                                    if result == False:
                                        discovered = False
                            if flag == True and discovered == True:
                                rfd_logger.info('[DISCOVERED] {0}, {1} --> {2}'.format(lhs1, lhs2, rhs))
                                discovered_rfd_logger.info('[DISCOVERED] {0}, {1} --> {2}'.format(lhs1, lhs2, rhs))
                                discovered_rfd_logger.info('{0}, {1} --> {2}'.format(feature_names[lhs1], feature_names[lhs2], feature_names[rhs]))


def discovery_three_to_one(neighbourhoods, feature_names):
    for lhs1 in range(0, len(neighbourhoods)):
        lhs1_neighbourhoods = neighbourhoods[lhs1]
        if (len(lhs1_neighbourhoods) == 0):
            rfd_logger.info('Feature {0} haven\'t neighbourhoods'.format(lhs1))
        else: 
            for lhs2 in range((lhs1+1), len(neighbourhoods)):
            # for lhs2k in range((lhs1), len(neighbourhoods)): #Testing i,i,i-->i
                for lhs3 in range((lhs2+1), len(neighbourhoods)):
                # for lhs3 in range((lhs2), len(neighbourhoods)): #Testing i,i,i-->i
                    for rhs in range(0, len(neighbourhoods)):
                        if rhs != lhs1 and rhs != lhs2 and rhs != lhs3:
                        # if rhs == lhs1 and rhs == lhs2 and rhs == lhs3: #Testing i,i,i-->i
                            rfd_logger.info('[TRYING] {0}, {1}, {2} --> {3}'.format(lhs1, lhs2, lhs3, rhs))
                            lhs2_neighbourhoods = neighbourhoods[lhs2]
                            lhs3_neighbourhoods = neighbourhoods[lhs3]
                            rhs_neighbourhoods = neighbourhoods[rhs]
                            if len(lhs2_neighbourhoods) > 0 and len(lhs3_neighbourhoods) > 0 and len(rhs_neighbourhoods) > 0:
                                flag = False
                                discovered = True
                                for ts in range(0, len(lhs1_neighbourhoods)):
                                    # ts_neighbourhood_of_lhs = list_operations.intersection_between_three(lhs1_neighbourhoods[ts], lhs2_neighbourhoods[ts], lhs3_neighbourhoods[ts])
                                    # ts_neighbourhood_of_rhs = rhs_neighbourhoods[ts]
                                    ts_neighbourhood_of_lhs = set(lhs1_neighbourhoods[ts]).intersection(set(lhs2_neighbourhoods[ts])).intersection(set(lhs3_neighbourhoods[ts]))
                                    ts_neighbourhood_of_rhs = set(rhs_neighbourhoods[ts])
                                    if (len(ts_neighbourhood_of_lhs) > 0 and len(ts_neighbourhood_of_rhs) > 0):
                                        flag = True
                                        # ts_neighbourhood_of_lhs.sort()
                                        # ts_neighbourhood_of_rhs.sort()
                                        # equals = np.array_equal(ts_neighbourhood_of_lhs, ts_neighbourhood_of_rhs)
                                        # print(len(ts_neighbourhood_of_lhs))
                                        # print(len(ts_neighbourhood_of_rhs))
                                        result = ts_neighbourhood_of_rhs.issuperset(ts_neighbourhood_of_lhs)
                                        if result == False:
                                            discovered = False
                                if flag == True and discovered == True:
                                    rfd_logger.info('[DISCOVERED] {0}, {1}, {2} --> {3}'.format(lhs1, lhs2, lhs3, rhs))
                                    discovered_rfd_logger.info('[DISCOVERED] {0}, {1}, {2} --> {3}'.format(lhs1, lhs2, lhs3, rhs))
                                    discovered_rfd_logger.info('{0}, {1}, {2} --> {3}'.format(feature_names[lhs1], feature_names[lhs2], feature_names[lhs3], feature_names[rhs]))


def discovery_four_to_one(neighbourhoods, feature_names):
    for lhs1 in range(0, len(neighbourhoods)):
        lhs1_neighbourhoods = neighbourhoods[lhs1]
        if (len(lhs1_neighbourhoods) == 0):
            rfd_logger.info('Feature {0} haven\'t neighbourhoods'.format(lhs1))
        else: 
            for lhs2 in range((lhs1+1), len(neighbourhoods)):
            # for lhs2 in range((lhs1), len(neighbourhoods)):  #Testing i,i,i,i-->i
                for lhs3 in range((lhs2+1), len(neighbourhoods)):
                # for lhs3 in range((lhs2), len(neighbourhoods)):   #Testing i,i,i,i-->i
                    for lhs4 in range((lhs3+1), len(neighbourhoods)):
                    # for lhs4 in range((lhs3), len(neighbourhoods)): #Testing i,i,i,i-->i
                        for rhs in range(0, len(neighbourhoods)):
                            if rhs != lhs1 and rhs != lhs2 and rhs != lhs3 and rhs != lhs4:
                            # if rhs == lhs1 and rhs == lhs2 and rhs == lhs3 and rhs == lhs4:   #Testing i,i,i,i-->i
                                rfd_logger.info('[TRYING] {0}, {1}, {2}, {3} --> {4}'.format(lhs1, lhs2, lhs3, lhs4, rhs))
                                lhs2_neighbourhoods = neighbourhoods[lhs2]
                                lhs3_neighbourhoods = neighbourhoods[lhs3]
                                lhs4_neighbourhoods = neighbourhoods[lhs4]
                                rhs_neighbourhoods = neighbourhoods[rhs]
                                if len(lhs2_neighbourhoods) > 0 and len(lhs3_neighbourhoods) > 0 and len(lhs4_neighbourhoods) > 0 and len(rhs_neighbourhoods) > 0:
                                    flag = False
                                    discovered = True
                                    for ts in range(0, len(lhs1_neighbourhoods)):
                                        # ts_neighbourhood_of_lhs = list_operations.intersection_between_four(lhs1_neighbourhoods[ts], lhs2_neighbourhoods[ts], lhs3_neighbourhoods[ts], lhs4_neighbourhoods[ts])
                                        # ts_neighbourhood_of_rhs = rhs_neighbourhoods[ts]
                                        ts_neighbourhood_of_lhs = set(lhs1_neighbourhoods[ts]).intersection(set(lhs2_neighbourhoods[ts])).intersection(set(lhs3_neighbourhoods[ts])).intersection(set(lhs4_neighbourhoods[ts]))
                                        ts_neighbourhood_of_rhs = set(rhs_neighbourhoods[ts])
                                        if (len(ts_neighbourhood_of_lhs) > 0 and len(ts_neighbourhood_of_rhs) > 0):
                                            flag = True
                                            # ts_neighbourhood_of_lhs.sort()
                                            # ts_neighbourhood_of_rhs.sort()
                                            # equals = np.array_equal(ts_neighbourhood_of_lhs, ts_neighbourhood_of_rhs)
                                            result = ts_neighbourhood_of_rhs.issuperset(ts_neighbourhood_of_lhs)
                                            if result == False:
                                                discovered = False
                                    if flag == True and discovered == True:
                                        rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3} --> {4}'.format(lhs1, lhs2, lhs3, lhs4, rhs))
                                        discovered_rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3} --> {4}'.format(lhs1, lhs2, lhs3, lhs4, rhs))
                                        discovered_rfd_logger.info('{0}, {1}, {2}, {3} --> {4}'.format(feature_names[lhs1], feature_names[lhs2], feature_names[lhs3], feature_names[lhs4], feature_names[rhs]))


def discovery_five_to_one(neighbourhoods, feature_names):
    for lhs1 in range(0, len(neighbourhoods)):
        lhs1_neighbourhoods = neighbourhoods[lhs1]
        if (len(lhs1_neighbourhoods) == 0):
            rfd_logger.info('Feature {0} haven\'t neighbourhoods'.format(lhs1))
        else: 
            for lhs2 in range((lhs1+1), len(neighbourhoods)):
            # for lhs2 in range((lhs1), len(neighbourhoods)):  #Testing i,i,i,i,i-->i
                for lhs3 in range((lhs2+1), len(neighbourhoods)):
                # for lhs3 in range((lhs2), len(neighbourhoods)):   #Testing i,i,i,i,i-->i
                    for lhs4 in range((lhs3+1), len(neighbourhoods)):
                    # for lhs4 in range((lhs3), len(neighbourhoods)): #Testing i,i,i,i,i-->i
                        for lhs5 in range((lhs4+1), len(neighbourhoods)):
                        # for lhs5 in range((lhs4), len(neighbourhoods)): #Testing i,i,i,i,i-->i
                            for rhs in range(0, len(neighbourhoods)):
                                if rhs != lhs1 and rhs != lhs2 and rhs != lhs3 and rhs != lhs4 and rhs != lhs5:
                                # if rhs == lhs1 and rhs == lhs2 and rhs == lhs3 and rhs == lhs4 and rhs == lhs5:   #Testing i,i,i,i,i-->i
                                    rfd_logger.info('[TRYING] {0}, {1}, {2}, {3}, {4} --> {5}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, rhs))
                                    lhs2_neighbourhoods = neighbourhoods[lhs2]
                                    lhs3_neighbourhoods = neighbourhoods[lhs3]
                                    lhs4_neighbourhoods = neighbourhoods[lhs4]
                                    lhs5_neighbourhoods = neighbourhoods[lhs5]
                                    rhs_neighbourhoods = neighbourhoods[rhs]
                                    if len(lhs2_neighbourhoods) > 0 and len(lhs3_neighbourhoods) > 0 and len(lhs4_neighbourhoods) > 0 and len(lhs5_neighbourhoods) > 0 and len(rhs_neighbourhoods) > 0:
                                        flag = False
                                        discovered = True
                                        for ts in range(0, len(lhs1_neighbourhoods)):
                                            ts_neighbourhood_of_lhs = set(lhs1_neighbourhoods[ts]).intersection(set(lhs2_neighbourhoods[ts])).intersection(set(lhs3_neighbourhoods[ts])).intersection(set(lhs4_neighbourhoods[ts])).intersection(set(lhs5_neighbourhoods[ts]))
                                            ts_neighbourhood_of_rhs = set(rhs_neighbourhoods[ts])
                                            if (len(ts_neighbourhood_of_lhs) > 0 and len(ts_neighbourhood_of_rhs) > 0):
                                                flag = True
                                                result = ts_neighbourhood_of_rhs.issuperset(ts_neighbourhood_of_lhs)
                                                if result == False:
                                                    discovered = False
                                        if flag == True and discovered == True:
                                            rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3}, {4} --> {5}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, rhs)) 
                                            discovered_rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3}, {4} --> {5}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, rhs))
                                            discovered_rfd_logger.info('{0}, {1}, {2}, {3}, {4} --> {5}'.format(feature_names[lhs1], feature_names[lhs2], feature_names[lhs3], feature_names[lhs4], feature_names[lhs5], feature_names[rhs])) 


def discovery_six_to_one(neighbourhoods, feature_names):
    for lhs1 in range(0, len(neighbourhoods)):
        lhs1_neighbourhoods = neighbourhoods[lhs1]
        if (len(lhs1_neighbourhoods) == 0):
            rfd_logger.info('Feature {0} haven\'t neighbourhoods'.format(lhs1))
        else: 
            for lhs2 in range((lhs1+1), len(neighbourhoods)):
            # for lhs2 in range((lhs1), len(neighbourhoods)):  #Testing i,i,i,i,i,i-->i
                for lhs3 in range((lhs2+1), len(neighbourhoods)):
                # for lhs3 in range((lhs2), len(neighbourhoods)):   #Testing i,i,i,i,i,i-->i
                    for lhs4 in range((lhs3+1), len(neighbourhoods)):
                    # for lhs4 in range((lhs3), len(neighbourhoods)): #Testing i,i,i,i,i,i-->i
                        for lhs5 in range((lhs4+1), len(neighbourhoods)):
                        # for lhs5 in range((lhs4), len(neighbourhoods)): #Testing i,i,i,i,i,i-->i
                            for lhs6 in range((lhs5+1), len(neighbourhoods)):
                            # for lhs6 in range((lhs5), len(neighbourhoods)): #Testing i,i,i,i,i,i-->i
                                for rhs in range(0, len(neighbourhoods)):
                                    if rhs != lhs1 and rhs != lhs2 and rhs != lhs3 and rhs != lhs4 and rhs != lhs5 and rhs != lhs6:
                                    # if rhs == lhs1 and rhs == lhs2 and rhs == lhs3 and rhs == lhs4 and rhs == lhs5 and rhs == lhs6:   #Testing i,i,i,i,i,i-->i
                                        rfd_logger.info('[TRYING] {0}, {1}, {2}, {3}, {4}, {5} --> {6}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, rhs))
                                        lhs2_neighbourhoods = neighbourhoods[lhs2]
                                        lhs3_neighbourhoods = neighbourhoods[lhs3]
                                        lhs4_neighbourhoods = neighbourhoods[lhs4]
                                        lhs5_neighbourhoods = neighbourhoods[lhs5]
                                        lhs6_neighbourhoods = neighbourhoods[lhs6]
                                        rhs_neighbourhoods = neighbourhoods[rhs]
                                        if len(lhs2_neighbourhoods) > 0 and len(lhs3_neighbourhoods) > 0 and len(lhs4_neighbourhoods) > 0 and len(lhs5_neighbourhoods) > 0 and len(lhs6_neighbourhoods) > 0 and len(rhs_neighbourhoods) > 0:
                                            flag = False
                                            discovered = True
                                            for ts in range(0, len(lhs1_neighbourhoods)):
                                                ts_neighbourhood_of_lhs = set(lhs1_neighbourhoods[ts]).intersection(set(lhs2_neighbourhoods[ts])).intersection(set(lhs3_neighbourhoods[ts])).intersection(set(lhs4_neighbourhoods[ts])).intersection(set(lhs5_neighbourhoods[ts])).intersection(set(lhs6_neighbourhoods[ts]))
                                                ts_neighbourhood_of_rhs = set(rhs_neighbourhoods[ts])
                                                if (len(ts_neighbourhood_of_lhs) > 0 and len(ts_neighbourhood_of_rhs) > 0):
                                                    flag = True
                                                    result = ts_neighbourhood_of_rhs.issuperset(ts_neighbourhood_of_lhs)
                                                    if result == False:
                                                        discovered = False
                                            if flag == True and discovered == True:
                                                rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3}, {4}, {5} --> {6}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, rhs))
                                                discovered_rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3}, {4}, {5} --> {6}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, rhs))
                                                discovered_rfd_logger.info('{0}, {1}, {2}, {3}, {4}, {5} --> {6}'.format(feature_names[lhs1], feature_names[lhs2], feature_names[lhs3], feature_names[lhs4], feature_names[lhs5], feature_names[lhs6], feature_names[rhs]))


def discovery_seven_to_one(neighbourhoods, feature_names):
    for lhs1 in range(0, len(neighbourhoods)):
        lhs1_neighbourhoods = neighbourhoods[lhs1]
        if (len(lhs1_neighbourhoods) == 0):
            rfd_logger.info('Feature {0} haven\'t neighbourhoods'.format(lhs1))
        else: 
            for lhs2 in range((lhs1+1), len(neighbourhoods)):
            # for lhs2 in range((lhs1), len(neighbourhoods)):  #Testing i,i,i,i,i,i,i-->i
                for lhs3 in range((lhs2+1), len(neighbourhoods)):
                # for lhs3 in range((lhs2), len(neighbourhoods)):   #Testing i,i,i,i,i,i,i-->i
                    for lhs4 in range((lhs3+1), len(neighbourhoods)):
                    # for lhs4 in range((lhs3), len(neighbourhoods)): #Testing i,i,i,i,i,i,i-->i
                        for lhs5 in range((lhs4+1), len(neighbourhoods)):
                        # for lhs5 in range((lhs4), len(neighbourhoods)): #Testing i,i,i,i,i,i,i-->i
                            for lhs6 in range((lhs5+1), len(neighbourhoods)):
                            # for lhs6 in range((lhs5), len(neighbourhoods)): #Testing i,i,i,i,i,i,i-->i
                                for lhs7 in range((lhs6+1), len(neighbourhoods)):
                                # for lhs7 in range((lhs6), len(neighbourhoods)): #Testing i,i,i,i,i,i,i-->i
                                    for rhs in range(0, len(neighbourhoods)):
                                        if rhs != lhs1 and rhs != lhs2 and rhs != lhs3 and rhs != lhs4 and rhs != lhs5 and rhs != lhs6 and rhs != lhs7:
                                        # if rhs == lhs1 and rhs == lhs2 and rhs == lhs3 and rhs == lhs4 and rhs == lhs5 and rhs == lhs6 and rhs == lhs7:   #Testing i,i,i,i,i,i,i-->i
                                            rfd_logger.info('[TRYING] {0}, {1}, {2}, {3}, {4}, {5}, {6} --> {7}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, lhs7, rhs))
                                            lhs2_neighbourhoods = neighbourhoods[lhs2]
                                            lhs3_neighbourhoods = neighbourhoods[lhs3]
                                            lhs4_neighbourhoods = neighbourhoods[lhs4]
                                            lhs5_neighbourhoods = neighbourhoods[lhs5]
                                            lhs6_neighbourhoods = neighbourhoods[lhs6]
                                            lhs7_neighbourhoods = neighbourhoods[lhs7]
                                            rhs_neighbourhoods = neighbourhoods[rhs]
                                            if len(lhs2_neighbourhoods) > 0 and len(lhs3_neighbourhoods) > 0 and len(lhs4_neighbourhoods) > 0 and len(lhs5_neighbourhoods) > 0 and len(lhs6_neighbourhoods) > 0 and len(lhs7_neighbourhoods) > 0 and len(rhs_neighbourhoods) > 0:
                                                flag = False
                                                discovered = True
                                                for ts in range(0, len(lhs1_neighbourhoods)):
                                                    ts_neighbourhood_of_lhs = set(lhs1_neighbourhoods[ts]).intersection(set(lhs2_neighbourhoods[ts])).intersection(set(lhs3_neighbourhoods[ts])).intersection(set(lhs4_neighbourhoods[ts])).intersection(set(lhs5_neighbourhoods[ts])).intersection(set(lhs6_neighbourhoods[ts])).intersection(set(lhs7_neighbourhoods[ts]))
                                                    ts_neighbourhood_of_rhs = set(rhs_neighbourhoods[ts])
                                                    if (len(ts_neighbourhood_of_lhs) > 0 and len(ts_neighbourhood_of_rhs) > 0):
                                                        flag = True
                                                        result = ts_neighbourhood_of_rhs.issuperset(ts_neighbourhood_of_lhs)
                                                        if result == False:
                                                            discovered = False
                                                if flag == True and discovered == True:
                                                    rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3}, {4}, {5}, {6} --> {7}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, lhs7, rhs))
                                                    discovered_rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3}, {4}, {5}, {6} --> {7}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, lhs7, rhs))
                                                    discovered_rfd_logger.info('{0}, {1}, {2}, {3}, {4}, {5}, {6} --> {7}'.format(feature_names[lhs1], feature_names[lhs2], feature_names[lhs3], feature_names[lhs4], feature_names[lhs5], feature_names[lhs6], feature_names[lhs7], feature_names[rhs]))


def discovery_eight_to_one(neighbourhoods, feature_names):
    for lhs1 in range(0, len(neighbourhoods)):
        lhs1_neighbourhoods = neighbourhoods[lhs1]
        if (len(lhs1_neighbourhoods) == 0):
            rfd_logger.info('Feature {0} haven\'t neighbourhoods'.format(lhs1))
        else: 
            for lhs2 in range((lhs1+1), len(neighbourhoods)):
            # for lhs2 in range((lhs1), len(neighbourhoods)):  #Testing i,i,i,i,i,i,i,i-->i
                for lhs3 in range((lhs2+1), len(neighbourhoods)):
                # for lhs3 in range((lhs2), len(neighbourhoods)):   #Testing i,i,i,i,i,i,i,i-->i
                    for lhs4 in range((lhs3+1), len(neighbourhoods)):
                    # for lhs4 in range((lhs3), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i-->i
                        for lhs5 in range((lhs4+1), len(neighbourhoods)):
                        # for lhs5 in range((lhs4), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i-->i
                            for lhs6 in range((lhs5+1), len(neighbourhoods)):
                            # for lhs6 in range((lhs5), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i-->i
                                for lhs7 in range((lhs6+1), len(neighbourhoods)):
                                # for lhs7 in range((lhs6), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i-->i
                                    for lhs8 in range((lhs7+1), len(neighbourhoods)):
                                    # for lhs8 in range((lhs7), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i-->i
                                        for rhs in range(0, len(neighbourhoods)):
                                            if rhs != lhs1 and rhs != lhs2 and rhs != lhs3 and rhs != lhs4 and rhs != lhs5 and rhs != lhs6 and rhs != lhs7 and rhs != lhs8:
                                            # if rhs == lhs1 and rhs == lhs2 and rhs == lhs3 and rhs == lhs4 and rhs == lhs5 and rhs == lhs6 and rhs == lhs7 and rhs == lhs8:   #Testing i,i,i,i,i,i,i,i-->i
                                                rfd_logger.info('[TRYING] {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7} --> {8}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, lhs7, lhs8, rhs))
                                                lhs2_neighbourhoods = neighbourhoods[lhs2]
                                                lhs3_neighbourhoods = neighbourhoods[lhs3]
                                                lhs4_neighbourhoods = neighbourhoods[lhs4]
                                                lhs5_neighbourhoods = neighbourhoods[lhs5]
                                                lhs6_neighbourhoods = neighbourhoods[lhs6]
                                                lhs7_neighbourhoods = neighbourhoods[lhs7]
                                                lhs8_neighbourhoods = neighbourhoods[lhs8]
                                                rhs_neighbourhoods = neighbourhoods[rhs]
                                                if len(lhs2_neighbourhoods) > 0 and len(lhs3_neighbourhoods) > 0 and len(lhs4_neighbourhoods) > 0 and len(lhs5_neighbourhoods) > 0 and len(lhs6_neighbourhoods) > 0 and len(lhs7_neighbourhoods) > 0 and len(lhs8_neighbourhoods) > 0 and len(rhs_neighbourhoods) > 0:
                                                    flag = False
                                                    discovered = True
                                                    for ts in range(0, len(lhs1_neighbourhoods)):
                                                        ts_neighbourhood_of_lhs = set(lhs1_neighbourhoods[ts]).intersection(set(lhs2_neighbourhoods[ts])).intersection(set(lhs3_neighbourhoods[ts])).intersection(set(lhs4_neighbourhoods[ts])).intersection(set(lhs5_neighbourhoods[ts])).intersection(set(lhs6_neighbourhoods[ts])).intersection(set(lhs7_neighbourhoods[ts])).intersection(set(lhs8_neighbourhoods[ts]))
                                                        ts_neighbourhood_of_rhs = set(rhs_neighbourhoods[ts])
                                                        if (len(ts_neighbourhood_of_lhs) > 0 and len(ts_neighbourhood_of_rhs) > 0):
                                                            flag = True
                                                            result = ts_neighbourhood_of_rhs.issuperset(ts_neighbourhood_of_lhs)
                                                            if result == False:
                                                                discovered = False
                                                    if flag == True and discovered == True:
                                                        rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7} --> {8}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, lhs7, lhs8, rhs))
                                                        discovered_rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7} --> {8}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, lhs7, lhs8, rhs))
                                                        discovered_rfd_logger.info('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7} --> {8}'.format(feature_names[lhs1], feature_names[lhs2], feature_names[lhs3], feature_names[lhs4], feature_names[lhs5], feature_names[lhs6], feature_names[lhs7], feature_names[lhs8], feature_names[rhs]))


def discovery_nine_to_one(neighbourhoods, feature_names):
    for lhs1 in range(0, len(neighbourhoods)):
        lhs1_neighbourhoods = neighbourhoods[lhs1]
        if (len(lhs1_neighbourhoods) == 0):
            rfd_logger.info('Feature {0} haven\'t neighbourhoods'.format(lhs1))
        else: 
            for lhs2 in range((lhs1+1), len(neighbourhoods)):
            # for lhs2 in range((lhs1), len(neighbourhoods)):  #Testing i,i,i,i,i,i,i,i,i-->i
                for lhs3 in range((lhs2+1), len(neighbourhoods)):
                # for lhs3 in range((lhs2), len(neighbourhoods)):   #Testing i,i,i,i,i,i,i,i,i-->i
                    for lhs4 in range((lhs3+1), len(neighbourhoods)):
                    # for lhs4 in range((lhs3), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i,i-->i
                        for lhs5 in range((lhs4+1), len(neighbourhoods)):
                        # for lhs5 in range((lhs4), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i,i-->i
                            for lhs6 in range((lhs5+1), len(neighbourhoods)):
                            # for lhs6 in range((lhs5), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i,i-->i
                                for lhs7 in range((lhs6+1), len(neighbourhoods)):
                                # for lhs7 in range((lhs6), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i,i-->i
                                    for lhs8 in range((lhs7+1), len(neighbourhoods)):
                                    # for lhs8 in range((lhs7), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i,i-->i
                                        for lhs9 in range((lhs8+1), len(neighbourhoods)):
                                        # for lhs9 in range((lhs8), len(neighbourhoods)): #Testing i,i,i,i,i,i,i,i,i-->i
                                            for rhs in range(0, len(neighbourhoods)):
                                                if rhs != lhs1 and rhs != lhs2 and rhs != lhs3 and rhs != lhs4 and rhs != lhs5 and rhs != lhs6 and rhs != lhs7 and rhs != lhs8 and rhs != lhs9:
                                                # if rhs == lhs1 and rhs == lhs2 and rhs == lhs3 and rhs == lhs4 and rhs == lhs5 and rhs == lhs6 and rhs == lhs7 and rhs == lhs8 and rhs == lhs9:   #Testing i,i,i,i,i,i,i,i,i-->i
                                                    rfd_logger.info('[TRYING] {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8} --> {9}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, lhs7, lhs8, lhs9, rhs))
                                                    lhs2_neighbourhoods = neighbourhoods[lhs2]
                                                    lhs3_neighbourhoods = neighbourhoods[lhs3]
                                                    lhs4_neighbourhoods = neighbourhoods[lhs4]
                                                    lhs5_neighbourhoods = neighbourhoods[lhs5]
                                                    lhs6_neighbourhoods = neighbourhoods[lhs6]
                                                    lhs7_neighbourhoods = neighbourhoods[lhs7]
                                                    lhs8_neighbourhoods = neighbourhoods[lhs8]
                                                    lhs9_neighbourhoods = neighbourhoods[lhs9]
                                                    rhs_neighbourhoods = neighbourhoods[rhs]
                                                    if len(lhs2_neighbourhoods) > 0 and len(lhs3_neighbourhoods) > 0 and len(lhs4_neighbourhoods) > 0 and len(lhs5_neighbourhoods) > 0 and len(lhs6_neighbourhoods) > 0 and len(lhs7_neighbourhoods) > 0 and len(lhs8_neighbourhoods) > 0 and len(lhs9_neighbourhoods) > 0 and len(rhs_neighbourhoods) > 0:
                                                        flag = False
                                                        discovered = True
                                                        for ts in range(0, len(lhs1_neighbourhoods)):
                                                            ts_neighbourhood_of_lhs = set(lhs1_neighbourhoods[ts]).intersection(set(lhs2_neighbourhoods[ts])).intersection(set(lhs3_neighbourhoods[ts])).intersection(set(lhs4_neighbourhoods[ts])).intersection(set(lhs5_neighbourhoods[ts])).intersection(set(lhs6_neighbourhoods[ts])).intersection(set(lhs7_neighbourhoods[ts])).intersection(set(lhs8_neighbourhoods[ts])).intersection(set(lhs9_neighbourhoods[ts]))
                                                            ts_neighbourhood_of_rhs = set(rhs_neighbourhoods[ts])
                                                            if (len(ts_neighbourhood_of_lhs) > 0 and len(ts_neighbourhood_of_rhs) > 0):
                                                                flag = True
                                                                result = ts_neighbourhood_of_rhs.issuperset(ts_neighbourhood_of_lhs)
                                                                if result == False:
                                                                    discovered = False
                                                        if flag == True and discovered == True:
                                                            rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8} --> {9}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, lhs7, lhs8, lhs9, rhs))
                                                            discovered_rfd_logger.info('[DISCOVERED] {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8} --> {9}'.format(lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, lhs7, lhs8, lhs9, rhs))
                                                            discovered_rfd_logger.info('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8} --> {9}'.format(feature_names[lhs1], feature_names[lhs2], feature_names[lhs3], feature_names[lhs4], feature_names[lhs5], feature_names[lhs6], feature_names[lhs7], feature_names[lhs8], feature_names[lhs9], feature_names[rhs]))


def sfd(df):
    n_samples = df.shape[0]
    n_features = df.shape[1]
    neighbourhoods = []
    for i in range(0, n_features):
        feature = df.iloc[:, i]
        # print(type(feature))
        list_values = feature.tolist()
        #print('Mean: {0}'.format(np.mean(list_values)))
        #print('Variance: {0}'.format(np.var(list_values)))
        outliers = list_operations.detect_outlier(list_values)
        # print(outliers)
        list_without_outliers = list_operations.difference(list_values, outliers)
        # print('{0} | {1}'.format(len(list_values), len(list_without_outliers)))
        epsilon = calculate_epsilon(list_without_outliers)
        # print('Epsiolon: {0}'.format(epsilon))
        feature_neighbourhood = []
        found = False;
        for j in range(0, n_samples):
            ts_neighbourhood = []
            for k in range(0, n_samples):
                # if (j != k and abs(list_values[j] - list_values[k]) <= epsilon):
                if abs(list_values[j] - list_values[k]) <= epsilon:
                    found = True;
                    ts_neighbourhood.append(k)
            # print('Number of neighbourhoods of {0}-{1}: {2}'.format(i, j, len(ts_neighbourhood)))
            feature_neighbourhood.append(ts_neighbourhood)
        if (found == False):
            neighbourhoods.append([])
        else:
            neighbourhoods.append(feature_neighbourhood)
    # print(neighbourhoods)

    discovery_one_to_one(neighbourhoods, list(df.columns))
    discovery_two_to_one(neighbourhoods, list(df.columns))
    discovery_three_to_one(neighbourhoods, list(df.columns))
    discovery_four_to_one(neighbourhoods, list(df.columns))
    discovery_five_to_one(neighbourhoods, list(df.columns))
    discovery_six_to_one(neighbourhoods, list(df.columns))
    discovery_seven_to_one(neighbourhoods, list(df.columns))
    discovery_eight_to_one(neighbourhoods, list(df.columns))
    discovery_nine_to_one(neighbourhoods, list(df.columns))

def main():

    datasets = [
    'ChlorineConcentration',
    'ECG200',
    'ECG5000',
    'FordA',
    'FordB',
    'PhalangesOutlinesCorrect',
    'RefrigerationDevices',
    'TwoLeadECG',
    'TwoPatterns' 
    ]

    for dataset in datasets:

        mcfs_df = pd.read_pickle('Pickle_rfd/MCFS/{0}.pkl'.format(dataset))
        corr_df = pd.read_pickle('Pickle_rfd/Corr/{0}.pkl'.format(dataset))

        rfd_logger.info('[STARTED] {0} on MCFS features'.format(dataset))
        discovered_rfd_logger.info('')
        discovered_rfd_logger.info('')
        discovered_rfd_logger.info('[STARTED] {0} on MCFS features'.format(dataset))
        sfd(mcfs_df)
        rfd_logger.info('[ENDED] {0} on MCFS features'.format(dataset))
        discovered_rfd_logger.info('[ENDED] {0} on MCFS features'.format(dataset))
        discovered_rfd_logger.info('')
        rfd_logger.info('[STARTED] {0} on CORRELATION features'.format(dataset))
        discovered_rfd_logger.info('[STARTED] {0} on CORRELATION features'.format(dataset))
        sfd(corr_df)
        rfd_logger.info('[ENDED] {0} on CORRELATION features'.format(dataset))
        discovered_rfd_logger.info('[ENDED] {0} on CORRELATION features'.format(dataset))
        discovered_rfd_logger.info('')
        discovered_rfd_logger.info('')

if __name__ == '__main__':
    main()