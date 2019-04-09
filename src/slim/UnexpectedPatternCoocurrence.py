import scipy.sparse
import numpy as np

import collections 

from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from slim import Slim


class UnexpectedPatternCoocurrence(BaseEstimator):
    '''
    Unsupervised Outlier Detection using Unexpected Pattern Co-Occurences (UPC).
    
    The anomaly score of each sample is called the Unexpected Pattern Co-Occurence 
    Score, alternatively known as the Beauty and Brains (BnB) Score.  Anomalous 
    samples are determined by finding unexpected feature combinations.  This 
    estimator can deal directly with categorical data without the need to
    perform one-hot encoding. 
    
    This implementation examines feature pairs as its item-set and scores each
    example based on the description in section 3.3 of the paper.
        
    Parameters
    ----------
    patterns : string, optional (default='both')
        - 'slim' - Use slim to mine patterns for considerations
        - 'pair' - Use all length-2 patterns
        - 'both' - Use slim plus all possible length-2 pattenrs
    
    transform_function : function, optional (default=None)
        - Function to transform the data.  If using scikit-learn >= 0.20,
            it is generally helpful to use KBinsDiscretizer on continuous features
            with encode='ordinal' and strategy='uniform' .
    
    na_value : number, optional (default=0)
        - Specify a value to ignore, in addition to None.
    
    method : string, optional (default='count')
        - only used when patterns='slim'
        - 'count' - lengths are calculating by counting the number of elements
        - 'gain' - lengths are calculated using the gain formula given in 
            section 3.3 in [1].
    
    stop_threshold : number, optional (defaut=None)
        - only used when patterns='slim'
        - Specify a percentage gain subsequent iterations must achieve before stopping
            (based on the gain achieved with the first iteration)
    
    prune_code_table : boolean, optional (default=True)
        - only used when patterns='slim'
        - prune code-table of those items that do not longer acheive gain?
    
    References
    ----------
    .. [1] Bertens, Roel, Vreeken, Jiles and Siebes, Arno (2017)
            Efficently Discovering Unexpected Pattern Co-Occurences. SDM17.
            
    .. [2] Bertens, Roel, Vreeken, Jiles and Siebes, Arno (2016, Feb)
           Beauty and Brains: Detecting Anomalous Pattern Co-Ocurrences.
           Technical Report 1512.07048v2.
    '''

    def __init__(self, patterns='both', transform_function=None, na_value=0, method="count", stop_threshold=None, prune_codetable=True):   
        self.patterns = patterns             
        self.method = method
        self.na_value = na_value
        self.stop_threshold = stop_threshold
        self.prune_codetable = prune_codetable
        self.transform_function = transform_function

    def fit(self, X, y=None):
        """Fit estimator.  This extracts the patterns
        to consider and counts the occurrences of each.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Accepts categorical attributes, for which
            use ``int`` to represent.

        Returns
        -------
        self : object
            Returns self.
        """
        
        if y is not None:
            y = check_array(y, accept_sparse=True, ensure_2d=False, allow_nd=True, ensure_min_samples=0, ensure_min_features=0)            
        
        X = check_array(X, accept_sparse=['csr'])
        if self.transform_function is not None:
            X = self.transform_function(X)
        
        self.slim_ = Slim.Slim(method=self.method, na_value=self.na_value, stop_threshold=self.stop_threshold, \
                               prune_code_table=self.prune_codetable)
        
        self.numExamples_ = X.shape[0]
        self.patternCounts_ = []
        self.featureIdByDesc_ = {}
        singletonMap = {}
        used_length_2 = set()
        
        if self.patterns != 'pair':
            self.slim_.fit(X)
            ct = self.slim_.codeTable_
        else:
            (totalElements, nextFeatureId, featureDescById, exampleIdsByfeatureId, featureIdByExampleIds) = self.loadData(X, y)  
            self.slim.featureDescById_ = featureDescById   
            ct = None        
        
            for f in ct:
                if f.leftSupport is None:                
                    singletonMap[f.featureId] = f
                elif f.support > 0:
                    if len(f.components)==2:
                        score = self.bnbScore(f.totalSupport, f.leftTotalSupport, f.rightTotalSupport)                
                    else:
                        score = self.bnbScore(f.support, f.leftTotalSupport, f.rightTotalSupport)
                        
                    fp = FeaturePair(f, None, f.support, score, f.components)   
                    self.patternCounts_.append(fp)
                    
                    if len(f.components) == 2:
                        l = sorted(list(f.components))
                        used_length_2.add((l[0], l[1]))
        
        # Not all length-2 pairs will exist in the data, but we should
        # consider all of them, regardless if Slim found them interesting.
        unused_length_2 = set()
        singletons = list(singletonMap.values())
        for (i, f1) in enumerate(singletons):
            for j in range(i+1, len(singletons)):
                f2 = singletons[j]
                left = f1 if f1.featureId < f2.featureId else f2
                right = f2 if f1.featureId < f2.featureId else f1
                key = (left.featureId, right.featureId)
                if key not in used_length_2:
                    unused_length_2.add(key)
        
        for (featureId, desc) in self.slim_.featureDescById_.items():
            self.featureIdByDesc_[desc] = featureId
        
        pattern2Counts = self.compute2PatternCounts(X)
        for (pattern, count) in pattern2Counts.items():
            key = (pattern[0], pattern[1])
            if key in unused_length_2:
                left = singletonMap[pattern[0]]
                right = singletonMap[pattern[1]]
                score = self.bnbScore(count, left.totalSupport, right.totalSupport)
                components = set([left.featureId, right.featureId])
                fp = FeaturePair(left, right, count, score, components)
                self.patternCounts_.append(fp)
        
        return self  
    
    def compute2PatternCounts(self, X):
        if(scipy.sparse.issparse(X)):
            return self.compute2PatternCountsSparse(X)
        else:
            return self.compute2PatternCountsDense(X)
                
    def compute2PatternCountsSparse(self, X):
        pattern2Counts = collections.Counter()
        for row in X:
            for (i, index) in enumerate(row.indices):
                data = row.data[i]                
                if data != self.na_value: 
                    featureId = self.featureIdByDesc_[(index, data)]                   
                    for i1 in range((i+1), len(row.indices)):
                        index1 = row.indices[i1]
                        data1 = row.data[i1]                    
                        if data1 != None and data1 != self.na_value:
                            featureId1 = self.featureIdByDesc_[(index1, data1)]
                            pattern2Counts[(featureId,featureId1)] += 1
        return pattern2Counts
    
    def compute2PatternCountsDense(self, X):
        pattern2Counts = collections.Counter()
        for row in X:
            for idx in range(0, row.shape[0]):
                data = row[idx]
                if data != None and data != self.na_value:    
                    featureId =  self.featureIdByDesc_[(idx, data)]               
                    for idx1 in range((idx+1), row.shape[0]):
                        data1 = row[idx1]                         
                        if data1 != None and data1 != self.na_value:
                            featureId1 = self.featureIdByDesc_[(idx1, data1)]   
                            pattern2Counts[(featureId,featureId1)] += 1
        return pattern2Counts
    
    def predict(self, X):
        """Predict if a particular sample is an outlier or not.  This computes
        a UPC/BnB score of each example and populates the "scores" and
        "maxScore" attributes.

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
        (scores, patterns, max_score) = self.computeScores(X)  
        mean = np.mean(scores)
        stddev = np.std(scores)
        confidence = self.cantelli(mean, stddev, max_score)
        probability = [self.cantelli(mean, stddev, x1) for x1 in scores]
        is_inlier = [1 if x1 > confidence else -1 for x1 in probability]
        
        return is_inlier
    
    def cantelli(self, mean, stddev, bnbScore):
        '''
        Cantelli's Inequality. see [1] sec 4.2
        '''
        
        if stddev ==0:
            return 0
        
        k = (bnbScore - mean) / stddev
        confidence = 1.0 / (1.0 + k**2)
        return confidence
    
    def decision_function(self, X):
        """maximum anomaly score of each pattern.

        The anomaly score of an input sample is computed as
        the maximum of each pattern's UPC (or BnB) score.


        Parameters
        ----------
        X : The training input samples. 

        Returns
        -------
        scores : array containing the anomaly score of the 
            input samples. The higher, the more abnormal.

        """              
        (scores, patterns, maxScore) = self.computeScores(X)
        return scores
         
    def computeScores(self, X): 
        """Same as decision_function, but also returns the most 
        anomalous pattern for each example.
        
        Parameters
        ----------
        X : The training input samples. 

        Returns
        -------
        (scores, patterns, maxScore) : The elements contain:
            - scores: an array with anomaly scores
            - patterns: an array of 2-element tuples with the 
                most anomalous pattern combination
            - maxScore: the maximum score among the examples        
        """
        if self.slim_ is None:
            raise ValueError("Must call 'fit' first.")
        
        X = check_array(X, accept_sparse=['csr'])
                
        scores = []
        patterns = []
        maxScore = 0
        
        for (i, row) in enumerate(X):
            rowMaxScore = 0
            rowMaxPattern = None
            
            if(scipy.sparse.issparse(row)):
                itemsInRow = self.computeRowItemsSparse(row)
            else:
                itemsInRow = self.computeRowItemsDense(row)  
            
            for fp in self.patternCounts_:
                if fp.components.issubset(itemsInRow):
                    if fp.score > rowMaxScore:                 
                        rowMaxScore = fp.score
                        rowMaxPattern = fp.pattern
                if len(itemsInRow) == 0:
                    break                
            
            scores.append(rowMaxScore)
            patterns.append(rowMaxPattern)
            if rowMaxScore > maxScore:
                maxScore = rowMaxScore                     
        return (scores, patterns, maxScore)    
    
    
    def computeRowItemsSparse(self, row):
        itemsInRow = set()
        for (i,idx) in enumerate(row.indices):
            elem = row.data[i]
            if elem is not None and elem != self.na_value:
                key = (idx, elem)
                featureId = self.featureIdByDesc_.get(key)
                if featureId is not None:
                    itemsInRow.add(featureId)
        return itemsInRow
    
    def computeRowItemsDense(self, row):
        itemsInRow = set()
        endIndex = row.shape[0] 
        for idx in range(0, endIndex):
            elem = row[idx]
            if elem is not None and elem != self.na_value:
                key = (idx, elem)
                featureId = self.featureIdByDesc_.get(key)
                if featureId is not None:
                    itemsInRow.add(featureId)
        return itemsInRow
                
    def bnbScore(self, support, leftSupport, rightSupport):
        pxy = support / self.numExamples_
        px = leftSupport / self.numExamples_
        py = rightSupport / self.numExamples_
        
        log2pxy = 0 if pxy == 0 else np.log2(pxy)
        log2px = 0 if px == 0 else np.log2(px)
        log2py = 0 if py == 0 else np.log2(py)
            
        # sec 3.3:  score = max of -log2(P(XY)) + log2(P(X) * P(Y))
        score = -1 * (log2pxy - log2px - log2py)        
                        
        return score
    
class FeaturePair:
    def __init__(self, left, right, support, score, components):
        self.left = left if right is None or right.featureId < left.featureId else right
        self.right = right if right is None or right.featureId < left.featureId else left
        self.support = support
        self.score = score
        self.components = components
        self.pattern = (left.featureId, ) if right is None else (left.featureId, right.featureId)
        
    def __lt__(self, that):
        val = self.score - that.score
        val = val if val != 0 else self.left.featureId - that.left.featureId
        return val < 0       
                         
    def __str__(self):
        return "[" + str(self.pattern) + " /support=" + str(self.support) + " /score=" + str(self.score) + "] " + str(self.components)
    
    
