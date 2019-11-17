import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np

'''
def intersection_between_two(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def intersection_between_three(lst1, lst2, lst3):
    lst4 = [value for value in lst1 if value in lst2 and value in lst3]
    return lst4

def intersection_between_four(lst1, lst2, lst3, lst4):
    lst5 = [value for value in lst1 if value in lst2 and value in lst3 and value in lst4]
    return lst5
'''

def detect_outlier(data):
    outliers = []
    sorted(data)
    # q1 = primo quartile, q3 = terzo quaritle
    q1, q3 = np.percentile(data, [25, 75])
    # iqr = scarto interquartile
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr) 
    upper_bound = q3 + (1.5 * iqr)
    for x in data:
        if x < lower_bound or x > upper_bound:
            outliers.append(x)
    return outliers

def difference(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3
