from sklearn.base import BaseEstimator, ClassifierMixin
from slim import Slim
import numpy as np
import scipy.sparse


class SlimClassifier(BaseEstimator, ClassifierMixin):
    '''
    Classifier using the SLIM compression algorithm.
    
    One SLIM compressor instance per n classes is created.  
    Each unseen example is compressed using each of n SLIM instances.
    The unseen example is assigned to that class which achieves the best
    compression.  See [1] section 6.3.
    
    Parameters
    ----------
    method : string, optional (default='count')
        - 'count' - lengths are calculating by counting the number of elements
        - 'gain' - lengths are calculated using the gain formula given in 
            section 3.3 in [1].
    
    na_value : number, optional (default=0)
        - Specify a value to ignore, in addition to None.
    
    Attributes
    ----------
    classes_ : A list of possible classes
    class_X_ : A list, each element is a subset of X with those examples having
        the class in "classes_" with the corresponding index.
    class_models_ : A list of SLIM instances corresponding to "classes_"    
        
    References
    ----------
    .. [1] K. Smets and J. Vreeken (2012) 
            SLIM: Directly mining description patterns. SDM12.
                
    '''


    def __init__(self, method="count", na_value=0):
        self.method_ = method
        self.na_value_ = na_value
    
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
            self.class_models_.append(Slim.Slim(method=self.method_, na_value=self.na_value_))
        
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
                transformedSize = self.matrixSize(transformed)
                if minSize is None or minSize > transformedSize:
                    minSize = transformedSize
                    minIndex = i
            predictions.append(minIndex)
        return np.array(predictions)    
    
    def matrixSize(self, mat):
        if scipy.sparse.issparse(mat):
            return len(mat.data) + len(mat.indices) + len(mat.indptr)
        else:
            return len(np.nonzero(mat)[0]) * 2 + mat.shape[0]
        
    
    

