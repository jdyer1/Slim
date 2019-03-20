from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y
from sklearn.multiclass import check_classification_targets
from slim import Slim
import numpy as np
import scipy.sparse
from astropy.wcs.docstrings import row


class SlimClassifier(BaseEstimator, ClassifierMixin):
    '''
    Classifier using the SLIM compression algorithm.
    
    One SLIM compressor instance per n classes is created.  
    Each unseen example is compressed using each of n SLIM instances.
    The unseen example is assigned to that class which achieves the best
    compression.  See [1] section 6.7.
    
    This classifier is best with datasets containing many categorical
    features.  As it determines class based on compressed size, data
    with little opportunity for compression (few features) will not
    classify well.  Continuous variables should be binned prior to use.
    
    Parameters
    ----------
    method : string, optional (default='count')
        - 'count' - lengths are calculating by counting the number of elements
        - 'gain' - lengths are calculated using the gain formula given in 
            section 3.3 in [1].
    
    na_value : number, optional (default=0)
        - Specify a value to ignore, in addition to None.
        
    transform_function : function, optional (derfault=0)
        - Function to transform the data.  If using scikit-learn >= 0.20,
            it is generally helpful to use KBinsDiscretizer on continuous features
            with encode='ordinal' and strategy='uniform' .
    
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


    def __init__(self, method="count", na_value=0, transform_function=None):
        self.method = method
        self.na_value = na_value
        self.transform_function = transform_function
    
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
        (X, y) = check_X_y(X, y, accept_sparse=['csr'])
        check_classification_targets(y)
        if self.transform_function is not None:
            X = self.transform_function(X)
        self.classes_ = np.unique(y)
        self.class_X_ = []
        self.class_models_ = []
        for c in self.classes_:
            hasC = y==c
            self.class_X_.append(X[hasC])
            self.class_models_.append(Slim.Slim(method=self.method, na_value=self.na_value))
        
        for i in range(0, len(self.class_X_)):
            model = self.class_models_[i]
            data = self.class_X_[i]
            if data.shape[0] > 0:            
                model.fit(data)
            else:
                self.class_models_[i] = None
            
        return self
      
    

    def predict(self, X):
        try:
            if self.classes_ is None:
                raise ValueError("fit must be called first.")
        except:
                raise ValueError("fit must be called first.")
        X = check_array(X, accept_sparse=['csr'])
        if self.transform_function is not None:
            X = self.transform_function(X)
        
        predictions = []
        for row in X:
            minSize = None
            minIndex = None
            row_transformed = row.reshape(1, -1)
            for i in range(0, len(self.class_models_)):
                model = self.class_models_[i]
                if model is not None:
                    transformed = model.transform(row_transformed)
                    transformedSize = self.matrixSize(transformed)
                    if minSize is None or minSize > transformedSize:
                        minSize = transformedSize
                        minIndex = i
            predictions.append(self.classes_[0] if minIndex is None else self.classes_[minIndex])
        return np.array(predictions)    
    
    def matrixSize(self, mat):
        if scipy.sparse.issparse(mat):
            return (len(np.nonzero(mat.data)[0]) * 2) + len(mat.indptr)
        else:
            return len(np.nonzero(mat)[0]) * 2 + mat.shape[0]
        
    
    def __copy__(self):
        return SlimClassifier(method=self.method, na_value=self.na_value, transform_function=self.transform_function)
    

