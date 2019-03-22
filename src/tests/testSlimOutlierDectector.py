import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import confusion_matrix
from slim import SlimOutlierDetector
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import KBinsDiscretizer
from timeit import default_timer as timer
import scipy.sparse
import os
import csv
import unittest

def test_compatibility():
    sc = SlimOutlierDetector.SlimOutlierDetector()
    check_estimator(sc)

def test_lympho():
    '''
    lymphography dataset - http://odds.cs.stonybrook.edu/lympho/
    '''
    X = loadData("lympho_X.csv")
    y = [1 if x==0 else -1 for x in loadData("lympho_y.csv").data]
    detect("lympho", False, X, y)    

def test_wbc():
    '''
    WBC dataset - http://odds.cs.stonybrook.edu/wbc/
    '''
    X = loadData("wbc_X.csv", dtype=np.float32)
    y = [1 if x==0 else -1 for x in loadData("wbc_y.csv").data]
    detect("wbc", True, X, y)    
 
    
def test_letter():
    '''
    Letter dataset - http://odds.cs.stonybrook.edu/letter-recognition-dataset/
    '''
    X = loadData("letter_X.csv")
    y = [1 if x==0 else -1 for x in loadData("letter_y.csv").data]
    detect("wbc", False, X, y) 

def test_glass():
    '''
    Glass dataset - http://odds.cs.stonybrook.edu/glass-data/
    '''
    X = loadData("glass_X.csv", dtype=np.float32)
    y = [1 if x==0 else -1 for x in loadData("glass_y.csv").data]
    detect("glass", True, X, y) 

def bin_data_function(X):
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    X = X.todense()
    return kbd.fit_transform(X)

def detect(label, bin_data, X, y):     
    for calc in ['length', 'difference', 'percent', 'forest']:    
        if calc=="forest":
            clf = IsolationForest(behaviour="new", contamination="auto")
        else:
            transform_function=None
            if bin_data: 
                transform_function=bin_data_function     
            clf = SlimOutlierDetector.SlimOutlierDetector(compress_eval_method=calc, transform_function=transform_function)
        
        is_inlier = clf.fit(X).predict(X)     
        correct = y == is_inlier
        numCorrect = np.sum(correct)
        numIncorrect = len(correct) - numCorrect
        (tn, fp, fn, tp) = confusion_matrix(y, is_inlier).ravel()             
        print(label, calc, numCorrect, numIncorrect, perc(numCorrect, numIncorrect), tn, fp, perc(tn, fp), fn, tp, perc(fn, tp), sep=",")
    
    #assert percCorrect > .97, "Less than 97% correct"

def perc(numCorrect, numIncorect):
    return '{:.2%}'.format(numCorrect / (numCorrect+numIncorect))

def loadData(csvFilename, dtype=np.int16):
    indptr = [0]
    indices = []
    data = []  
    
    filePath = os.path.join(os.path.dirname(__file__), csvFilename)
    with open(filePath, mode="r") as infile:
        r = csv.reader(infile)
        for row in r: 
            i = 0           
            for elem in row:                
                indices.append(i)
                data.append(elem)
                i += 1
            indptr.append(len(indices))                
    
    return scipy.sparse.csr_matrix((data, indices, indptr), dtype=dtype)

def logTime(label, start, totalTime):
    end = timer()
    used = end - start
    totalTime += used
    print(label + " time used", used, "total so far", totalTime)
    start = end   
    return (start, totalTime)

if __name__ == '__main__':
    print('label', 'calc', 'numCorrect', 'numIncorrect', '%', 'tn', 'fp', '%', 'fn', 'tp', '%', sep=",")
    test_lympho()
    test_wbc()
    test_letter()
    test_glass()
    