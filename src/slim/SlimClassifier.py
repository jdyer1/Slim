from sklearn.base import BaseEstimator, ClassifierMixin
from slim import Slim
import numpy as np
import scipy.sparse
import io


class SlimClassifier(BaseEstimator, ClassifierMixin):
    '''
    classdocs
    '''


    def __init__(self, method="count"):
        self.method = method
    
    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """   
        self.classes_, y = np.unique(y, return_inverse=True)
        self.class_X_ = []
        self.class_models_ = []
        for c in self.classes_:
            hasC = y==c
            self.class_X_.append(X[hasC])
            self.class_models_.append(Slim.Slim(self.method))
        
        for i in range(0, len(self.class_X_)):
            model = self.class_models_[i]
            data = self.class_X_[i]
            model.fit(data)  
            
        return self
      
    

    def predict(self, X):    
        predictions = []
        for row in X:
            minSize = None
            minIndex = None
            for i in range(0, len(self.class_models_)):
                model = self.class_models_[i]
                transformed = model.transform(row.reshape(1, -1))
                transformedSize = matrixSize(transformed)
                if minSize is None or minSize > transformedSize:
                    minSize = transformedSize
                    minIndex = i
            predictions.append(minIndex)
        return np.array(predictions)    
    
    #D = self.decision_function(X)
    #return self.classes_[np.argmax(D, axis=1)]

def matrixSize(mat):
    if scipy.sparse.issparse(mat):
        return len(mat.data) + len(mat.indices) + len(mat.indptr)
    else:
        return len(np.nonzero(mat)[0]) * 2 + mat.shape[0]
    
    
    

