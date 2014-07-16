#cython: profile=True 
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
from cython.operator cimport dereference as deref, preincrement as inc 
import cython
import struct
import numpy 
cimport numpy
import scipy.sparse 
numpy.import_array()
from sandbox.util.CythonUtils cimport dot, scale, choice, inverseChoice, uniformChoice, plusEquals


class MCEvaluatorCython(object):
    @staticmethod 
    def recommendAtk(numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, unsigned int k, X): 
        """
        Compute the matrix Z = U V^T and then find the k largest indices for each row 
        but exclude those in X. 
        """
        cdef unsigned int m = U.shape[0]          
        cdef numpy.ndarray[numpy.int32_t, ndim=2, mode="c"] orderedItems = numpy.zeros((U.shape[0], k), numpy.int32)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] scores = numpy.zeros(V.shape[0], numpy.float)
        cdef unsigned int i = 0
        cdef unsigned int itemInd 
        cdef double minScore 
        
        for i in range(m): 
            scores = U[i, :].dot(V.T)
            minScore = numpy.min(scores)-1           
            itemInd = 0
            
            while itemInd != k: 
                maxInd = numpy.argmax(scores)
                scores[maxInd] = minScore
                
                if X[i, maxInd] != 1: 
                    orderedItems[i, itemInd] = maxInd
                    itemInd += 1
                    
        return orderedItems 
        
    @staticmethod  
    def precisionAtk(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[int, ndim=2] indices): 
        """
        Take a list of nonzero indices, and also a list of predicted indices and compute 
        the precision. 
        """
        cdef unsigned int i, j
        cdef double count
        cdef unsigned int k = indices.shape[1]
        cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] precisions = numpy.zeros(indices.shape[0], numpy.float)
        
        for i in range(indices.shape[0]):
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            count = 0 
            for j in range(k): 
                if indices[i, j] in omegai: 
                    count += 1
            precisions[i] = count/k
        
        return precisions
            
    @staticmethod 
    def recallAtk(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[int, ndim=2] indices): 
        """
        Take a list of nonzero indices, and also a list of predicted indices and compute 
        the recall. 
        """
        cdef unsigned int i, j
        cdef double count
        cdef unsigned int k = indices.shape[1]
        cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] recalls = numpy.zeros(indices.shape[0], numpy.float)
        
        for i in range(indices.shape[0]):
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            count = 0 
            for j in range(k): 
                if indices[i, j] in omegai: 
                    count += 1
            if omegai.shape[0] != 0: 
                recalls[i] = count/omegai.shape[0]
        
        return recalls  

    @staticmethod 
    def stratifiedRecallAtk(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[int, ndim=2] indices, numpy.ndarray[int, ndim=1] itemCounts, double beta): 
        """
        Take a list of nonzero indices, and also a list of predicted indices and compute 
        the stratified recall as given in Steck, Item Popularity and Recommendation 
        Accuracy, 2011. 
        """
        cdef unsigned int i, j
        cdef double numerator
        cdef unsigned int k = indices.shape[1]
        cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] recalls = numpy.zeros(indices.shape[0], numpy.float)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] denominators = numpy.zeros(indices.shape[0], numpy.float)
        
        for i in range(indices.shape[0]):
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            numerator = 0 
            denominator = 0
            
            for j in range(k): 
                if indices[i, j] in omegai: 
                    numerator += 1/itemCounts[indices[i,j]]**beta
                    
            for j in omegai:
                denominators[i] +=  1/itemCounts[j]**beta                   
                    
            if denominators[i] != 0: 
                recalls[i] = numerator/denominators[i]
        
        return recalls, denominators  

    @staticmethod
    def reciprocalRankAtk(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[int, ndim=2] indices): 
        """
        Take a list of nonzero indices, and also a list of predicted indices and compute 
        the reciprocal rank at k. 
        """
        cdef unsigned int i, j
        cdef unsigned int k = indices.shape[1]
        cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] rrs = numpy.zeros(indices.shape[0], numpy.float)
        
        for i in range(indices.shape[0]):
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            count = 0 
            for j in range(k): 
                if indices[i, j] in omegai: 
                    rrs[i] = 1/float(1+j)
                    break
        
        return rrs  
        
    @staticmethod
    def localAUCApprox(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[int, ndim=1, mode="c"] allIndPtr, numpy.ndarray[int, ndim=1, mode="c"] allColInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, unsigned int numAucSamples, numpy.ndarray[double, ndim=1, mode="c"] r): 
        """
        Compute the estimated local AUC for the score functions UV^T relative to a matrix 
        X with quantile vector r. If evaluating on a set of test observations then (indPtr, colInds)
        is the test set and (allIndPtr, allColInds) is all positive elements. 
        """
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0]
        cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[int, ndim=1, mode="c"] allOmegai 
        cdef numpy.ndarray[int, ndim=1, mode="c"] omegaiSample
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] localAucArr = numpy.zeros(m)
        cdef unsigned int i, j, k, ind, p, q, nOmegai
        cdef double partialAuc, ri, uivp
    
        k = U.shape[1]
    
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            allOmegai = allColInds[allIndPtr[i]:allIndPtr[i+1]]
            nOmegai = omegai.shape[0]
            ri = r[i]
            
            if nOmegai * (n-nOmegai) != 0: 
                partialAuc = 0    
                omegaiSample = uniformChoice(omegai, numAucSamples)  
                #omegaiSample = numpy.random.choice(omegai, numAucSamples)
                #omegaiSample = omegai
                
                for p in omegaiSample:                
                    q = inverseChoice(allOmegai, n)                
                    uivp = dot(U, i, V, p, k)
    
                    if uivp > ri and uivp > dot(U, i, V, q, k): 
                        partialAuc += 1 
                            
                localAucArr[i] = partialAuc/float(omegaiSample.shape[0])     
        
        return localAucArr.mean()

