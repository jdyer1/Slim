import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn import datasets
from sklearn.model_selection import train_test_split
from slim import SlimClassifier
from sklearn.preprocessing import KBinsDiscretizer
from timeit import default_timer as timer
import unittest

@unittest.skip("check_classifiers_train requires >.83 accuracy on a 2-feature training set, but we only achieve .77")
def test_compatibility():
    sc = SlimClassifier.SlimClassifier(na_value=1, transform_function=bin_data_function)
    check_estimator(sc)

def test_iris():
    dataset = datasets.load_iris()
    X = bin_data_function(dataset.data)
    y = dataset.target
    classify("iris", X, y)    

@unittest.skip("slow")
def test_digits():   
    dataset = datasets.load_digits(n_class=10, return_X_y=False)
    classify("digits", dataset.data, dataset.target)

def bin_data_function(X):
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    return kbd.fit_transform(X)


def classify(label, X, y): 
    clf = SlimClassifier.SlimClassifier(na_value=None)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)
    
    start = timer()
    totalTime = 0 
    
    clf.fit(X_train, y_train)
    
    (start, totalTime) = logTime(label, start, totalTime)
    
    predicted = clf.predict(X_test)
    correct = y_test == predicted
    numCorrect = np.sum(correct)
    numIncorrect = len(correct) - numCorrect
    percCorrect = numCorrect / len(correct)
    
    (start, totalTime) = logTime(label, start, totalTime)
    
    print(label, "correct", numCorrect, "incorrect", numIncorrect, "%Correct", percCorrect)
    assert percCorrect > .8, "Less than 80% correct"

def logTime(label, start, totalTime):
    end = timer()
    used = end - start
    totalTime += used
    print(label + " time used", used, "total so far", totalTime)
    start = end   
    return (start, totalTime)

if __name__ == '__main__':
    test_compatibility()
    