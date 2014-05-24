# cython: profile=True
from cython.operator cimport dereference as deref, preincrement as inc 
import cython
import struct
import numpy 
cimport numpy
import scipy.sparse 
numpy.import_array()


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
        the precision. 
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