import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn import datasets
from sklearn.model_selection import train_test_split
from slim import SlimClassifier
from sklearn.preprocessing import KBinsDiscretizer
from timeit import default_timer as timer
import unittest


def test_iris():
    dataset = datasets.load_iris()
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    X = kbd.fit_transform(dataset.data)
    y = dataset.target
    classify("iris", X, y)    

@unittest.skip("slow")
def test_digits():   
    dataset = datasets.load_digits(n_class=10, return_X_y=False)
    classify("digits", dataset.data, dataset.target)

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
    
    print(label, "correct", numCorrect, "incorrect", numIncorrect, "percentCorrect", percCorrect)
    assert percCorrect > .8, "Less than 80% correct"

def logTime(label, start, totalTime):
    end = timer()
    used = end - start
    totalTime += used
    print(label + " time used", used, "total so far", totalTime)
    start = end   
    return (start, totalTime)

if __name__ == '__main__':
    test_iris()
    test_digits()
    
    