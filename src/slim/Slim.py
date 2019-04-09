import warnings
import copy
import numpy as np
import scipy.sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

class Slim(BaseEstimator, TransformerMixin):
    '''
    Compression transformer that finds descriptive patterns using the SLIM algorithm.
    
    SLIM builds a code table iteratively, identifying the best itemsets using 
    the Minimum Descriptive Length (MDL) principle.  
    
    Parameters
    ----------
    method : string, optional (default='count')
        - 'count' - lengths are calculating by counting the number of elements
        - 'gain' - lengths are calculated using the gain formula given in 
            section 3.3 in [1].
    
    na_value : number, optional (default=0)
        - Specify a value to ignore, in addition to None.
        
    stop_threshold : number, optional (defaut=None)
        - Specify a percentage gain subsequent iterations must achieve before stopping
            (based on the gain achieved with the first iteration)
    
    prune_code_table : boolean, optional (default=True)
        - prune code-table of those items that do not longer acheive gain?
    
    Attributes
    ----------
    codeTable_ : A list of Feature obects containing:
            - featureId : the internal feature id describing: (column in X, value in X)
            - cardinality : the number of signleton features this feature replaces.
            - components : a set of singleton feature Ids this feature replaces.
            - exampleIds : a set of row numbers in X that have this feature
            - standardCodeLength : the computed length of this code table entry
            - standardDatabaseLength : the computed length of the data described 
                by this code table entry
    
    featureDescById_ : A dictionary mapping feature id's to 2-tuples of:
        (column in X, value in X)
    
    References
    ----------
    .. [1] K. Smets and J. Vreeken (2012) 
            SLIM: Directly mining description patterns. SDM12.
                
    '''
    def __init__(self, method="count", na_value=0, stop_threshold=None, prune_code_table=True):
        method = "count" if method is None else method
        self.cl_ = LengthCalculator(method=method)
        self.na_value_ = na_value
        self.stop_threshold_ = stop_threshold
        self.prune_code_table_ = prune_code_table
        
        self.totalFeatures_ = None
        self.totalElements_ = 0   
        self.nextFeatureId_ = 0     
        self.databaseLength_ = 0
        self.codeTableSize_ = 0
        self.numUsedFeatures_ = 0 
        self.numberEvaluatedCandidates_ = 0
        
        self.dtype_ = None
        
        self.featureDescById_ = None
        self.codeTable_ = None
        self.candidates_ = None     
    
    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=['csr'])
        self.totalFeatures_ = X.shape[1]
        (totalElements, nextFeatureId, featureDescById, exampleIdsByfeatureId, featureIdByExampleIds) = self.loadData(X, y)
        
        self.dtype_ = X.dtype
        self.totalElements_ = totalElements
        self.nextFeatureId_ = nextFeatureId
        self.featureDescById_ = featureDescById   
        
        codeTable = []
        databaseLength = 0
        codeTableSize = 0
        
        for (featureId, exampleIds) in exampleIdsByfeatureId.items():
            
            feature = Feature(featureId, 1,set([featureId]))
            feature.exampleIds = exampleIds
            feature.originalExampleIds = exampleIds
            feature.support = len(exampleIds)
            feature.totalSupport = feature.support
            
            (feature.standardCodeLength, feature.databaseLength) = \
                self.cl_.computeLengths(feature.support, feature.cardinality, self.totalElements_)
                
            databaseLength += feature.databaseLength
            codeTableSize += feature.standardCodeLength
            
            codeTable.append(feature)         
        
        self.codeTable_ = codeTable
        self.databaseLength_ = databaseLength 
        self.codeTableSize_ = codeTableSize
        self.numUsedFeatures_ = len(codeTable) 
        
        self.candidates_ = []
        self.generateInitialCandidates() 
        self.candidates_ = sorted(self.candidates_)
        self.numberEvaluatedCandidates_ = 0
        
        iterationNum = 1
        firstIterGain = 0
                    
        while len(self.candidates_) > 0:     
            candidate = self.candidates_[0]
            del self.candidates_[0]
            
            candidate.computeStatistics(self.totalElements_)
            self.numberEvaluatedCandidates_ += 1
            if candidate.estimatedGain <= 0:
                accept = True
            else:
                accept = False
                        
            if not accept:
                continue    
            
            candidate.apply()
            newFeature = self.addNewFeature(candidate)
            
            new_candidates = []
            thisIterGain = 0
            for candidate1 in self.candidates_:
                if candidate1.i_feature.support == 0 or candidate1.j_feature.support == 0:
                    continue
                
                if candidate1.i_feature == candidate.i_feature or \
                 candidate1.i_feature == candidate.j_feature or \
                 candidate1.j_feature == candidate.i_feature or \
                 candidate1.j_feature == candidate.j_feature:
                
                    candidate1.computeStatistics(self.totalElements_) 
                    
                if candidate1.estimatedGain < 0:
                    new_candidates.append(candidate1)
                    thisIterGain += candidate1.estimatedGain
            
            if self.stop_threshold_ is not None:    
                if iterationNum==1:
                    firstIterGain = abs(thisIterGain * self.stop_threshold_)            
                elif abs(thisIterGain) < firstIterGain:
                    break
            
            self.candidates_ = sorted(new_candidates)              
            
            iterationNum += 1
            self.generateCandidates(newFeature) 
        
        if self.prune_code_table_:
            self.pruneCodeTable()
        self.codeTable_ = sorted(self.codeTable_)
        
        return self
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.do_transform()
    
    def transform(self, X): 
        X = check_array(X, accept_sparse=['csr'])
        if X.shape[1] != self.totalFeatures_:
            raise ValueError("matrix to transform should have the same features as fit matrix: " + str(self.totalFeatures_))
        
        (totalElements, nextFeatureId, featureDescById, exampleIdsByfeatureId, featureIdByExampleIds) = self.loadData(X, None, True)
        
        
        codeTable = []
        codeTableFeatureIds = set()
        for feature in self.codeTable_:
            f1 = copy.copy(feature)
            codeTable.append(f1)   
            codeTableFeatureIds.add(f1.featureId)
        
        for featureId in exampleIdsByfeatureId:
            if featureId not in codeTableFeatureIds:  
                codeTable.append(Feature(featureId, 1, set([featureId])))
        
        for (exampleId, featureIds) in featureIdByExampleIds.items():
            for f in codeTable:
                if f.components.issubset(featureIds):
                    featureIds = featureIds.difference(f.components)
                    f.exampleIds.add(exampleId)
                    f.support = f.support + 1
                    f.totalSupport = f.support
                       
                
        return self.do_transform(codeTable)
    
    def do_transform(self, codeTable=None):
        featureIdsByExampleIds = {}
        maxExampleId = -1
        
        ct = self.codeTable_ if codeTable is None else codeTable
        
        unusedFeatureIds = set(self.featureDescById_)
        for f in ct: 
            for exampleId in f.exampleIds:
                maxExampleId = max(maxExampleId, exampleId)
                featureIds = featureIdsByExampleIds.setdefault(exampleId, [])
                featureIds.append(f.featureId)
                unusedFeatureIds.discard(f.featureId)
                
        if len(unusedFeatureIds) == 0:
            unusedFeatureIds = None       
        
        indptr = [0]
        indices = []
        data = []  
        for i in range(0,maxExampleId + 1):              
            featureIds = featureIdsByExampleIds.setdefault(i, [])
            for featureId in featureIds:
                indices.append(featureId)
                data.append(1)
            
            # This is to pass sklearn/utils/estimator_checks._apply_on_subsets
            # cf, https://github.com/scikit-learn/scikit-learn/commit/989f9c764734efc21bbff21b0202b52a1112bfc3
            if unusedFeatureIds is not None and i==0:
                for unusedFeatureId in unusedFeatureIds:
                    indices.append(unusedFeatureId)
                    data.append(0) 
                                       
            indptr.append(len(indices))
        if len(data) == 0:
            mat = scipy.sparse.csr_matrix((0,0))
        else:
            mat = scipy.sparse.csr_matrix((data, indices, indptr), dtype=self.dtype_)        
        return mat
                
    def get_code_table(self):
        codeTable = self.codeTable_
        
        if codeTable is None:
            raise(ValueError("'fit' or 'fit_transform' must first be called to generate the code table"))  
        
        return list((f.featureId, f.support, f.components) for f in codeTable)
    
    def get_feature_cross_reference(self):
        featureDescById = self.featureDescById_
        
        if featureDescById is None:
            raise(ValueError("'fit' or 'fit_transform' must first be called to generate the feature cross reference"))
        
        return dict(featureDescById)
    
    def loadData(self, X, y=None, returnXref=True):
        if y is not None:
            y = check_array(y, accept_sparse=True, ensure_2d=False, allow_nd=True, ensure_min_samples=0, ensure_min_features=0)
            if len(y)>0:
                warnings.warn("This estimator works on unlabelled data; Y is ignored", RuntimeWarning)
                
        if(scipy.sparse.issparse(X)):
            return self.loadDataSparse(X, returnXref)
        else:
            return self.loadDataDense(X, returnXref)        
    
    def loadDataSparse(self, X, returnXref=False):        
        (featureDescById, featureIdByDesc) = self.initializeFeatureCrossReference()
        exampleIdsByfeatureId = {}
        featureIdByExampleIds = {} if returnXref else None
        nextFeatureId = self.nextFeatureId_
        
        for (rowId, row) in enumerate(X):
            for (i, idx) in enumerate(row.indices):
                element = row.data[i]
                if element is not None and element != self.na_value_:
                    featureDesc = (idx, element)                
                    featureId = featureIdByDesc.get(featureDesc)
                    if featureId is None:
                        featureId = nextFeatureId
                        nextFeatureId += 1
                        featureIdByDesc[featureDesc] = featureId
                        featureDescById[featureId] = featureDesc
                    
                    exampleIdsByfeatureId.setdefault(featureId, set()).add(rowId)
                    
                    if featureIdByExampleIds is not None:
                        featureIdByExampleIds.setdefault(rowId, set()).add(featureId)
                
        totalElements = len(X.data)
        return (totalElements, nextFeatureId, featureDescById, exampleIdsByfeatureId, featureIdByExampleIds)
    
    
    def loadDataDense(self, X, returnXref=False): 
        (featureDescById, featureIdByDesc) = self.initializeFeatureCrossReference()
        exampleIdsByfeatureId = {} 
        featureIdByExampleIds = {} if returnXref else None
        nextFeatureId = self.nextFeatureId_
        totalElements = 0
        
        for (rowId, row) in enumerate(X):
            for (idx, element) in enumerate(row):
                if element is not None and element != self.na_value_:
                    totalElements += 1
                    featureDesc = (idx, element) 
                    featureId = featureIdByDesc.get(featureDesc)
                    if featureId is None:
                        featureId = nextFeatureId
                        nextFeatureId += 1
                        featureIdByDesc[featureDesc] = featureId
                        featureDescById[featureId] = featureDesc
                
                    exampleIdsByfeatureId.setdefault(featureId, set()).add(rowId)
                    
                    if featureIdByExampleIds is not None:
                        featureIdByExampleIds.setdefault(rowId, set()).add(featureId)
                
        return (totalElements, nextFeatureId, featureDescById, exampleIdsByfeatureId, featureIdByExampleIds)
    
    def initializeFeatureCrossReference(self):
        if self.featureDescById_ is None:
            featureDescById = {}
            featureIdByDesc = {}
        else:
            featureDescById = self.featureDescById_.copy()
            featureIdByDesc = {}
            for (featureId, featureDesc) in featureDescById.items():
                featureIdByDesc[featureDesc] = featureId
            
        return (featureDescById, featureIdByDesc)
    
    def generateInitialCandidates(self):
        self.codeTable_ = sorted(self.codeTable_)
        for (i, i_feature) in enumerate(self.codeTable_):
            self.generateCandidates(i_feature, i)            
    
    def generateCandidates(self, i_feature, i=None):    
        startAt = 0 if i is None else i+1     
        for j in range(startAt,len(self.codeTable_)):
            j_feature = self.codeTable_[j]
            if i_feature.featureId == j_feature.featureId:
                continue
            
            if i_feature.support == 0 or j_feature.support == 0:
                continue
            
            if len(i_feature.components.intersection(j_feature.components)) > 0:
                continue
            
            candidate = Candidate(self.cl_, i_feature, j_feature, self.totalElements_)
            
            if candidate.estimatedGain < 0:
                self.candidates_.append(candidate) 
    
    def addNewFeature(self, candidate):                
        components = set(candidate.i_feature.components).union(candidate.j_feature.components)        
        cardinality = len(components)
        f = Feature(self.nextFeatureId_, cardinality, components)
        f.leftTotalSupport = candidate.i_feature.totalSupport
        f.rightTotalSupport = candidate.j_feature.totalSupport
        f.leftSupport = candidate.i_feature.support
        f.rightSupport = candidate.j_feature.support
        f.exampleIds = candidate.exampleIds
        f.standardCodeLength = candidate.standardCodeLength           
        f.support = candidate.support 
        if candidate.i_feature.originalExampleIds is None or candidate.j_feature.originalExampleIds is None:
            f.totalSupport = f.support
        else:
            f.totalSupport = len(candidate.i_feature.originalExampleIds.intersection(candidate.j_feature.originalExampleIds))  
        f.databaseLength = candidate.databaseLength
        
        self.nextFeatureId_ += 1
        
        self.codeTable_.append(f)
        self.totalElements_ = candidate.new_totalElements
        self.codeTableSize_ = self.codeTableSize_ + candidate.gain_codeTableSize 
        self.databaseLength_ = self.databaseLength_ + candidate.gain_databaseLength
        
        return f
    
    def pruneCodeTable(self):
        numberNonsingletonElements = 0
        allComponents = set()
        for feature in self.codeTable_:                
            allComponents = allComponents.union(feature.components)
            
        newCodeTable = []
        for feature in self.codeTable_:
            if feature.support > 0 or feature.featureId in allComponents:
                newCodeTable.append(feature)
                if len(feature.components) > 1:
                    numberNonsingletonElements += 1
        self.codeTable_ = newCodeTable  
        self.numberNonsingletonElements_ = numberNonsingletonElements
                
        
    def decompress(self, transformed=None, code_table=None, feature_cross_reference=None):
        if transformed is None:
            if code_table is not None:
                warnings.warn("When 'transformed' is unspecified, the internal code table is used. 'code_table' is ignored.", RuntimeWarning)
            if feature_cross_reference is not None:
                warnings.warn("When 'transformed' is unspecified, the internal cross reference is used. 'feature_cross_reference' is ignored.", RuntimeWarning)
            
            originalFeaturesByExampleId = {}
        
            indptr = [0]
            indices = []
            data = []   
            
            for feature in self.codeTable_:
                for exampleId in feature.exampleIds:
                    componentSet = originalFeaturesByExampleId.get(exampleId)
                    if componentSet is None:
                        componentSet = set()
                        originalFeaturesByExampleId[exampleId] = componentSet
                    for component in feature.components:
                        componentSet.add(self.featureDescById_[component])
                    
            exampleIds = sorted(list(originalFeaturesByExampleId.keys()))
            for rowId in exampleIds:
                for (colId, val) in sorted(list(originalFeaturesByExampleId[rowId])):
                    indices.append(colId)
                    data.append(val)
                indptr.append(len(indices))
            mat = scipy.sparse.csr_matrix((data, indices, indptr), dtype=self.dtype_)
            return mat
        else:
            if code_table is None:
                code_table = self.get_code_table()
            if feature_cross_reference is None:
                feature_cross_reference = self.get_feature_cross_reference()
                            
            indptr = [0]
            indices = []
            data = []
            
            codeTableByFeatureId = {}
            for (featureId, support, components) in code_table:
                codeTableByFeatureId[featureId] = components
            
            for (exampleId, example) in enumerate(transformed):
                allComponents = set()
                
                for i in range(0, len(example.indices)):
                    featureId = example.indices[i]
                    if example.data[i] == 1:         
                        allComponents.update(codeTableByFeatureId[featureId])
                
                tuples = []
                for componentId in allComponents:
                    idxAndVal = feature_cross_reference[componentId]
                    tuples.append(idxAndVal)
                for idxAndVal in sorted(tuples, key=lambda x: x[0]):
                    indices.append(idxAndVal[0])
                    data.append(idxAndVal[1])
                indptr.append(len(indices))
            
            mat = scipy.sparse.csr_matrix((data, indices, indptr), dtype=self.dtype_)
            return mat


def matrixSize(mat):
    '''
    callable from outside class instance
    '''
    if scipy.sparse.issparse(mat):
        return (len(np.nonzero(mat.data)[0]) * 2) + len(mat.indptr)
    else:
        return len(np.nonzero(mat)[0]) * 2 + mat.shape[0]


class LengthCalculator:
    def __init__(self, method):
        if method == 'count':
            self.gain = False
        elif method == 'gain':
            self.gain = True
        else:
            raise(ValueError("unrecognized method: " + method))
            
    
    def computeLengths(self, support, cardinality, totalDatabaseElements):
        if self.gain:
            codeLength = 0 if support==0 or totalDatabaseElements==0 else -np.log2(support/totalDatabaseElements)
            databaseLength = support * codeLength
            return (codeLength, databaseLength)
        else:
            return (cardinality, support)
            
class Candidate:
    def __init__(self, cl, i_feature, j_feature, totalElements):
        self.cl = cl
        self.i_feature = i_feature
        self.j_feature = j_feature
        self.computeStatistics(totalElements, True)
        
    def computeStatistics(self, totalElements, force=False):
        if not force and self.totalElements == totalElements:
            return
        
        self.totalElements = totalElements
                
        self.exampleIds = self.i_feature.exampleIds.intersection(self.j_feature.exampleIds)
        self.support = len(self.exampleIds)
        
        self.new_totalElements = totalElements - self.support
              
        self.cardinality = self.i_feature.cardinality + self.j_feature.cardinality
        (self.standardCodeLength, self.databaseLength) = self.cl.computeLengths(self.support, self.cardinality, self.new_totalElements)
        
        self.new_i_support = self.i_feature.support - self.support
        self.new_j_support = self.j_feature.support - self.support
        
        (self.new_i_standardCodeLength, self.new_i_databaseLength) = \
            self.cl.computeLengths(self.new_i_support, self.i_feature.cardinality, self.new_totalElements)
        (self.new_j_standardCodeLength, self.new_j_databaseLength) = \
            self.cl.computeLengths(self.new_j_support, self.j_feature.cardinality, self.new_totalElements)    
        
        self.gain_codeTableSize = \
            self.standardCodeLength \
            - self.i_feature.standardCodeLength + self.new_i_standardCodeLength \
            - self.j_feature.standardCodeLength + self.new_j_standardCodeLength
        
        self.gain_databaseLength = \
            self.databaseLength \
            - self.i_feature.databaseLength + self.new_i_databaseLength \
            - self.j_feature.databaseLength + self.new_j_databaseLength
        
        self.estimatedGain = (self.gain_codeTableSize + self.gain_databaseLength)
    
    def apply(self):
        self.i_feature.support = self.new_i_support
        self.i_feature.standardCodeLength = self.new_i_standardCodeLength
        self.i_feature.databaseLength = self.new_i_databaseLength
        self.i_feature.exampleIds = self.i_feature.exampleIds.difference(self.exampleIds)
        
        self.j_feature.support = self.new_j_support
        self.j_feature.standardCodeLength = self.new_j_standardCodeLength
        self.j_feature.databaseLength = self.new_j_databaseLength
        self.j_feature.exampleIds = self.j_feature.exampleIds.difference(self.exampleIds)
            
    def __lt__(self, that):
        val = self.estimatedGain - that.estimatedGain
        val = val if val != 0 else self.i_feature.featureId - that.i_feature.featureId
        val = val if val != 0 else self.j_feature.featureId - that.j_feature.featureId
        return val < 0
    
    def __hash__(self):
        return hash(self.i_feature.featureId) + hash(self.j_feature.featureId)
    
    def __eq__(self, that):
        return True \
            if self.i_feature.featureId == that.i_feature.featureId \
            and self.j_feature.featureId == that.j_feature.featureId \
            else False
    
    def __str__(self):
        return str(self.i_feature.components) + ", " + str(self.j_feature.components) + ": "  + \
            str(self.estimatedGain) + " (" + str(self.gain_databaseLength) + " + " + \
            str(self.gain_codeTableSize) + ")"

class Feature:
    def __init__(self, featureId, cardinality, components=set()):
        self.featureId = featureId
        self.cardinality = cardinality
        self.components = components
        self.exampleIds = set()
        self.originalExampleIds = None
        self.standardCodeLength = 0 
        self.support = 0
        self.totalSupport = 0
        self.leftSupport = None
        self.rightSupport = None
        self.leftTotalSupport = None
        self.rightTotalSupport = None
        self.databaseLength = 0 
            
    def __lt__(self, that):
        # Standard Cover Order, sec 2.3
        val =  that.cardinality - self.cardinality
        val = val if val != 0 else that.support - self.support
        val = val if val != 0 else self.featureId - that.featureId
        return val < 0
    
    def __hash__(self):
        return hash(self.featureId)
    
    def __eq__(self, that):
        return True if self.featureId == that.featureId else False
    
    def __copy__(self):
        return Feature(self.featureId, self.cardinality, self.components)
                            
    def __str__(self):
        return "[featureId: " + str(self.featureId) + " /support: " + str(self.support) + \
            " /totalSupport: " + str(self.totalSupport)+ " /components: " + str(self.components) + "]"
    
   
    
    
        