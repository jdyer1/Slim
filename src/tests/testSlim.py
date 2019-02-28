import os
import csv
import re
import numpy as np
import scipy.sparse
from slim import Slim
from sklearn.utils.estimator_checks import check_estimator
from timeit import default_timer as timer


def test_compatibility():
    check_estimator(Slim.Slim())

def test_slim_sparse_count():
    csr = loadData("iris.csv")
    runSlim("sparse csr 'count' ", csr, "count")

def test_slim_dense_count():
    csr = loadData("iris.csv")
    runSlim("dense 'count' ", csr.todense(), "count")

def test_slim_sparse_gain():
    csr = loadData("iris.csv")
    runSlim("sparse csr 'gain'", csr, "gain")

def test_slim_dense_gain():
    csr = loadData("iris.csv")
    runSlim("dense 'gain'", csr.todense(), "gain")
    
def test_compare_with_reference_implementation_count():
    compareRefImpl("count")

def test_compare_with_reference_implementation_gain():
    compareRefImpl("gain")

def runSlim(desc, X, method):    
        start = timer()
        totalTime = 0    
        
        s = Slim.Slim(method)
        compressed = s.fit_transform(X)
        (start, totalTime) = logTime(desc + " fit-transform", start, totalTime)   
        
        originalSize = matrixSize(X) 
        codeTable = s.get_code_table()     
        xref = s.get_feature_cross_reference()        
        totalSize = compute_total_size(s, compressed)
        print("original size", originalSize, "total compressed size", totalSize) 
        (start, totalTime) = logTime(desc + " return codetable & xref", start, totalTime)        
        
        assert totalSize < originalSize, "compressed size is not less than original: " + desc        
        assert isCodeTableInStandardCoverOrder(codeTable), "code table not in standard cover order."
        
        decompressed = s.decompress() 
        (start, totalTime) = logTime(desc + " decompress method 1", start, totalTime)   
        assert matrixEquals(X, decompressed), "decompressed (method 1) is not the same as the original: " + desc
        
        decompressed = s.decompress(compressed, codeTable, xref)
        (start, totalTime) = logTime(desc + " decompress method 2", start, totalTime)   
        assert matrixEquals(X, decompressed), "decompressed (method 2) is not the same as the original: " + desc
        
        compressed1 = s.transform(X)
        (start, totalTime) = logTime(desc + " transform", start, totalTime)
        decompressed = s.decompress(compressed1)
        (start, totalTime) = logTime(desc + " decompress method 3", start, totalTime)
        assert matrixEquals(X, decompressed), "decompressed (method 3) is not the same as the original: " + desc

def compareRefImpl(method):
    '''
    This test compares the code table produced by the reference implementation "slimMJ"
    
    We first check that the code table produced by this implementation is at least as good
    at compressing as "slimMJ".  
    
    Then we ensure that both code tables decompress back to the original.
    '''
    
    X = loadData("iris.csv")
    s = Slim.Slim(method=method)    
    
    '''
    First we produce a compressed, and re-decompressed dataset 
    using our implementation's code table
    '''    
    compressed = s.fit_transform(X)
    decompressed = s.decompress(compressed)
    totalSize = compute_total_size(s, compressed)
    
    
    '''
    We re-load the data to obtain "exampleIdsByFeatureId" which the implementation discards
    '''
    (totalElements, nextFeatureId, featureDescById, exampleIdsByfeatureId) = s.loadData(X)
    
    '''
    Now we load the SlimMJ code table, being careful to translate 
    the feature id's used their to our feature id's.
    '''    
    featureIdXref = {}
    for (featureId, featureDesc) in featureDescById.items():
        featureIdXref[featureDesc[0]] = featureId            
   
    refImplCodeTable = loadReferenceImplCodeTable("iris-ct-slimmj-gain.txt", featureIdXref, exampleIdsByfeatureId)
    
    s.codeTable_ = refImplCodeTable
    assert isCodeTableInStandardCoverOrder(s.get_code_table()), "ref impl code table not in standard cover order."    
    
    '''
    Next we compress then re-decompress the data using the SlimMJ code table.
    '''
    compressed1 = s.do_transform()
    totalSize1 = compute_total_size(s, compressed1)
    decompressed1 = s.decompress(compressed1)
    
    '''
    Finally compare the two results
    '''    
    assert totalSize < totalSize1, "our compression is worse than ref impl compression."    
    assert matrixEquals(X, decompressed), "decompressed (ours) is not the same as the original" 
    assert matrixEquals(X, decompressed1), "decompressed (ref impl) is not the same as the original" 
    assert matrixEquals(decompressed, decompressed1), "decompressed (ours) is not the same as decompressed (ref impl)" 

def compute_total_size(slim, compressedMatrix):
    codeTable = slim.get_code_table()
    xref = slim.get_feature_cross_reference()
    compressedSize = matrixSize(compressedMatrix)
    codetableSize = codeTableSize(codeTable)
    xrefSize = len(xref) * 3
    totalSize = compressedSize + codetableSize + xrefSize
    return totalSize

def loadData(csvFilename):
    indptr = [0]
    indices = []
    data = []  
    
    filePath = os.path.join(os.path.dirname(__file__), csvFilename)
    with open(filePath, mode="r") as infile:
        r = csv.reader(infile)
        for row in r:            
            for elem in row:
                if(len(elem) > 0):
                    indices.append(int(elem))
                    data.append(1)
            indptr.append(len(indices))                
    
    return scipy.sparse.csr_matrix((data, indices, indptr), dtype=np.int8)

def loadReferenceImplCodeTable(slimMjCodeTableFilename, featureIdXref, exampleIdsByfeatureId):
    '''
    This is the code table generated by the reference implementation.
    See here: http://eda.mmci.uni-saarland.de/prj/upc/
    
    Settings used:
    
    algo = slimMJ-cccoverpartial-usg
    command = 2D
    datatype = bm128
    eststrategy = gain
    internalmineto = memory
    iscchunktype = isc
    iscifmined = zap
    iscname = iris-all-1d
    iscstoretype = isc
    maxtime = 0
    preferredafopt = internal
    prunestrategy = pep
    takeiteasy = 0
    taskclass = anomaly
    thresholdbits = 0
    '''
    p = re.compile(r"^([\d\s]+)[(](\d+)[,]\d+[)]\s*$")
    filePath = os.path.join(os.path.dirname(__file__), slimMjCodeTableFilename)
    with open(filePath, mode="r") as f:
        singletons = []
        multiples = []
        for a in f:
            result = p.match(a)
            if result:
                spaceDelimitedComponents = result.group(1)
                support = int(result.group(2))
                components = set()
                for component in spaceDelimitedComponents.split():
                    componentId = int(component)
                    xrefComponentId = featureIdXref[componentId]
                    components.add(xrefComponentId)
                entry = (components, support)
                if len(components)==1:
                    singletons.append(entry)
                else:
                    multiples.append(entry)
                
        features = []
        maxFeatureId = -1
        for entry in singletons:
            
            assert len(entry[0]) == 1            
            featureId = -1
            for fid in entry[0]:
                featureId = fid
                
            f = Slim.Feature(featureId, 1, entry[0])
            f.support = entry[1]
            features.append(f)
            maxFeatureId = max(maxFeatureId, featureId)
            
        featureId = maxFeatureId + 1
        
        for entry in multiples:
            f = Slim.Feature(featureId, len(entry[0]), entry[0])
            f.support = entry[1]
            features.append(f)
            featureId += 1   
        
        codeTable = sorted(features)
        
        for f in codeTable:
            examples = None
            for c in f.components:            
                exampleIds = exampleIdsByfeatureId.get(c)
                if examples is None:
                    examples = exampleIds
                else:
                    examples = examples.intersection(exampleIds)
                    
            f.exampleIds = examples
            
            for c in f.components:
                exampleIds = exampleIdsByfeatureId.get(c)
                exampleIds = exampleIds.difference(examples)
                exampleIdsByfeatureId[c] = exampleIds   
                   
    return sorted(features)
            

def logTime(label, start, totalTime):
        end = timer()
        used = end - start
        totalTime += used
        print(label + " time used", used, "total so far", totalTime)
        start = end   
        return (start, totalTime)     

def matrixSize(mat):
    if scipy.sparse.issparse(mat):
        return len(mat.data) + len(mat.indices) + len(mat.indptr)
    else:
        return len(np.nonzero(mat)[0]) * 2 + mat.shape[0]

def isCodeTableInStandardCoverOrder(ct):
    lastFeatureId = None
    lastSupport = None
    lastCardinality = None
    seenFeatureIds = set()
    for (featureId, support, components) in ct:
        cardinality = len(components)
        if lastFeatureId is not None:   
            if featureId in seenFeatureIds:
                return False            
            if lastCardinality == cardinality:
                if lastSupport < support:
                    return False
                if lastSupport == support:
                    if lastFeatureId > featureId:
                        return False
            
        lastFeatureId = featureId 
        lastSupport = support
        lastCardinality = cardinality
        seenFeatureIds.add(featureId)
    return True

def codeTableSize(ct):
    size = 0
    for f in ct:
        size += 2
        size += len(f[2])
    return size

def matrixEquals(m1, m2):
    if scipy.sparse.issparse(m1):
        m1 = m1.todense()
        m2 = m2.todense()
    comp = (m1 == m2)
    if isinstance(comp, (list, np.ndarray)):
        return comp.all()
    return comp

