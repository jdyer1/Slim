import os
import csv

import numpy as np
import scipy.sparse

from sklearn.utils.estimator_checks import check_estimator
from sklearn import datasets
from slim import UnexpectedPatternCoocurrence
from sklearn.preprocessing import KBinsDiscretizer
from timeit import default_timer as timer

def test_compatibility():
    sc = UnexpectedPatternCoocurrence.UnexpectedPatternCoocurrence(na_value=0, transform_function=bin_data_function)
    check_estimator(sc)

def test_iris_slim():
    dataset = datasets.load_iris()
    X = bin_data_function(dataset.data)
    detect("iris", X, 'slim') 

def test_iris_pair():
    dataset = datasets.load_iris()
    X = bin_data_function(dataset.data)
    detect("iris", X, 'pair') 

def test_iris_both():
    dataset = datasets.load_iris()
    X = bin_data_function(dataset.data)
    detect("iris", X, 'both') 
    
def bin_data_function(X):
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    return kbd.fit_transform(X)


def detect(label, X, bin_data=True, patterns='both'): 
    '''
    TODO: needs some asserts.
    '''
    bin_function = bin_data_function if bin_data else None
    clf = UnexpectedPatternCoocurrence.UnexpectedPatternCoocurrence(transform_function=bin_function, patterns=patterns)    
    start = timer()
    totalTime = 0 
    
    is_inlier = clf.fit(X).predict(X)
    
    (start, totalTime) = logTime(label, start, totalTime)
    
def load_data_matrix(fileName):
    indptr = [0]
    indices = []
    data = []
    filePath = os.path.join(os.path.dirname(__file__), fileName)
    
    with open(filePath, mode="r") as f:
        r = csv.reader(f, delimiter=",")
        
        for row in r:
            for index in row:
                indices.append(index)
                data.append(1)
                
            indptr.append(len(indices))  
                      
    mat = scipy.sparse.csr_matrix((data, indices, indptr), dtype=np.int16)
    return mat
       

def logTime(label, start, totalTime):
    end = timer()
    used = end - start
    totalTime += used
    print(label + " time used", used, "total so far", totalTime)
    start = end   
    return (start, totalTime)

    