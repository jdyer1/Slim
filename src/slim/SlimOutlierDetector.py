import scipy.sparse
import numpy as np

from slim import Slim

from sklearn.base import BaseEstimator
from sklearn.utils import check_array

class SlimOutlierDetector(BaseEstimator):
    '''
    Unsupervised Outlier Detection using SLIM.
           
    Parameters
    ----------
    method : string, optional (default='count')
        - 'count' - lengths are calculating by counting the number of elements
        - 'gain' - lengths are calculated using the gain formula given in 
            section 3.3 in [1].
    
    na_value : number, optional (default=0)
        - Specify a value to ignore, in addition to None.
    
    compress_eval_method : string, optional (default='length')
        - 'length' - Compression is evaluated by the length of the compressed example
        - 'difference' - Compression is evaluated by the original length minus the compressed length
        - 'percent' - Compression is evaluated by the percent change from original to compressed.
        - Note: the definition of 'length' is governed by the 'method' parameter.  See above.
        
    transform_function : function, optional (default=None)
        - Function to transform the data.  If using scikit-learn >= 0.20,
            it is generally helpful to use KBinsDiscretizer on continuous features
            with encode='ordinal' and strategy='uniform' .

    References
    ----------
    .. [1] K. Smets and J. Vreeken (2012) 
            SLIM: Directly mining description patterns. SDM12.
    
    .. [2] K. Smets and J. Vreeken (2011)
            The odd one out: Identifying and characterising anomalies. SIAM 2011.
    '''

    def __init__(self, method="count", na_value=0, compress_eval_method="length", transform_function=None):
        self.method = method
        self.na_value = na_value
        self.compress_eval_method = compress_eval_method
        self.transform_function = transform_function
          
            
    def fit(self, X, y=None):  
        X = check_array(X, accept_sparse=['csr'])
        
        if self.transform_function is not None:
            X = self.transform_function(X)
        
        self.slim_ = Slim.Slim(method=self.method, na_value=self.na_value)
        compressed = self.slim_.fit_transform(X)        
        lengthByExample = self.lengthByRow(compressed, X)
        
        self.mean_ = np.mean(lengthByExample)
        self.stddev_ = np.std(lengthByExample)
        maxLength = np.max(lengthByExample)
        self.confidence_ = self.cantelli(maxLength)
                
        return self
    
    def cantelli(self, length):
        '''
        Cantelli's Inequality. see [2] sec 2.4
        '''
        if self.stddev_ == 0:
            return 0
        
        k = (length - self.mean_) / self.stddev_
        confidence = 1.0 / (1.0 + k**2)
        return confidence
        
    
    def lengthByRow(self, compressed, uncompressed):
        sizesUncompressed = (uncompressed != 0).sum(1)
        sizesCompressed = (compressed != 0).sum(1)
        length = []
        
        if self.compress_eval_method != "length" \
            and self.compress_eval_method != "difference" \
            and self.compress_eval_method != "percent":
                raise(ValueError("unrecognized compress_eval_method: " + self.compress_eval_method))
            
        
        rawLength = True if self.compress_eval_method=="length" else False
        percentGain = True if self.compress_eval_method=="percent" else False
        
        for i in range(0, len(sizesCompressed)):
            rawSize = sizesCompressed.item(i)
            sizeCompressed = self.slim_.cl_.computeLengths(rawSize, rawSize, self.slim_.totalElements_)[0]
            if rawLength:
                length.append(sizeCompressed)
            else:
                rawSize = sizesUncompressed.item(i) 
                sizeUncompressed = self.slim_.cl_.computeLengths(rawSize, rawSize, self.slim_.totalElements_)[0]
                gain = sizeUncompressed - sizeCompressed
                if percentGain:                    
                    percentCompression = gain / sizeUncompressed
                    length.append(percentCompression)
                else:
                    length.append(gain)            
        return length
    
    def decision_function(self, X):
        try:
            if self.slim_ is None:
                pass
        except:
                raise ValueError("fit must be called first.")
        
        X = check_array(X, accept_sparse=['csr'])        
        if self.transform_function is not None:
            X = self.transform_function(X)
        
        compressed = self.slim_.transform(X)
        lengthByRow = self.lengthByRow(compressed, X)
        scores = [self.cantelli(length) for length in lengthByRow]
        return np.array(scores)    
    
    def predict(self, X):
        """Predict if a particular sample is an outlier or not.  

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. 

        Returns
        -------
        is_inlier : array, shape (n_samples,)
            For each observations, tells whether or not (+1 or -1) it should
            be considered as an inlier.
        """    
        
        scores = self.decision_function(X)
        is_inlier = np.ones(len(scores), dtype=int)
        for i in range(0, len(scores)):
            is_inlier[i] = 1 if scores[i] >= self.confidence_ else -1
        return is_inlier

    
   
    
        