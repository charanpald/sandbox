# cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
from cython.operator cimport dereference as deref, preincrement as inc 
import cython
import struct
import numpy 
cimport numpy
import scipy.sparse 
from sandbox.util.Util import Util
from sandbox.util.CythonUtils cimport uniformChoice
numpy.import_array()

cdef extern from "SparseUtilsCython.cpp": 
    void partialReconstructValsPQCpp(int*, int*, double*, double*, double*, int, int) 

class SparseUtilsCython(object): 
    """
    Some Cythonised functions for sparse matrices. 
    """
        
    @staticmethod 
    def partialReconstructValsPQ(numpy.ndarray[int, ndim=1] rowInds, numpy.ndarray[int, ndim=1] colInds, numpy.ndarray[double, ndim=2, mode="c"] P, numpy.ndarray[double, ndim=2, mode="c"] Q): 
        """
        Given an array of unique indices inds, partially reconstruct $P*Q^T$. Do 
        the heavy work in C++. 
        """ 
        if P.shape[1] != Q.shape[1]: 
            raise ValueError("Matrices not aligned")
        
        cdef numpy.ndarray[double, ndim=1, mode="c"] values = numpy.zeros(rowInds.shape[0])
        partialReconstructValsPQCpp(&rowInds[0], &colInds[0], &P[0,0], &Q[0,0], &values[0], rowInds.shape[0], P.shape[1])          
        return values        

    @staticmethod 
    def partialReconstructPQ(omega, P, Q): 
        """
        Given an array of unique indices inds, partially reconstruct $P*Q^T$.
        The returned matrix is a scipy csc_matrix.
        """ 
        rowInds = numpy.array(omega[0], numpy.int32)
        colInds = numpy.array(omega[1], numpy.int32)
        vals = SparseUtilsCython.partialReconstructValsPQ(rowInds, colInds, P, Q)
        inds = numpy.c_[omega[0], omega[1]].T
        X = scipy.sparse.csc_matrix((vals, inds), shape=(P.shape[0], Q.shape[0]))
        
        return X      
     
    @staticmethod
    def partialOuterProduct(numpy.ndarray[int, ndim=1] rowInds, numpy.ndarray[int, ndim=1] colInds, numpy.ndarray[numpy.float_t, ndim=1] u, numpy.ndarray[numpy.float_t, ndim=1] v):
        """
        Given an array of unique indices omega, partially reconstruct a matrix 
        using two vectors u and v 
        """ 
        cdef unsigned int i
        cdef unsigned int j 
        cdef unsigned int k
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] values = numpy.zeros(rowInds.shape[0], numpy.float)
        
        for i in range(rowInds.shape[0]):
            j = rowInds[i]
            k = colInds[i]
            
            values[i] = u[j]*v[k]            
            
        return values    
    
    @staticmethod
    def sumCols(numpy.ndarray[int, ndim=1] rowInds, numpy.ndarray[unsigned char, ndim=1] vals, int numRows): 
        """
        Take the sum along each row of the given matrix and reduce overflow issues by storing the results in a float array. 
        """
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] result = numpy.zeros(numRows, numpy.float)
        cdef unsigned int i
    
        for i in range(rowInds.shape[0]): 
            result[rowInds[i]] += vals[i]

        return result 
        
    @staticmethod
    @cython.profile(False)  
    def computeR(numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, double w, unsigned int indsPerRow=50): 
        """
        Given a matrix Z = UV.T compute a vector r such r[i] is the uth quantile 
        of the ith row of Z. We sample indsPerRow elements in each row and use that 
        for computing quantiles. Thus u=0 implies the smallest element and u=1 implies 
        the largest. 
        """
        cdef int m = U.shape[0]
        cdef int n = V.shape[0]
        #indsPerRow = min(indsPerRow, n)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] r = numpy.zeros(m, numpy.float)
        cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] tempRows = numpy.zeros((m, indsPerRow), numpy.float)
        cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] colInds = numpy.zeros(indsPerRow, numpy.int)

        colInds = numpy.random.choice(n, indsPerRow, replace=True)
        tempRows = U.dot(V[colInds, :].T)
        r = numpy.percentile(tempRows, w*100.0, 1)
        
        return r  
        
        
    @staticmethod
    def computeR2(numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] w, unsigned int indsPerRow=50): 
        """
        Given a matrix Z = UV.T compute a vector r such r[i] is the uth quantile 
        of the ith row of Z. We sample indsPerRow elements in each row and use that 
        for computing quantiles. Thus u=0 implies the smallest element and u=1 implies 
        the largest. 
        """
        cdef int m = U.shape[0]
        cdef int n = V.shape[0]
        #indsPerRow = min(indsPerRow, n)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] r = numpy.zeros(m, numpy.float)
        cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] tempRows = numpy.zeros((m, indsPerRow), numpy.float)
        cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] colInds = numpy.zeros(indsPerRow, numpy.int)

        colInds = numpy.random.choice(n, indsPerRow, replace=True)
        tempRows = U.dot(V[colInds, :].T)
        
        for i in range(m): 
            r[i] = numpy.percentile(tempRows[i, :], w[i]*100.0)
        
        return r  
    
    @staticmethod
    def centerRowsCsarray(X):
        """
        Center the nonzero elements of X by row. 
        """
        cdef unsigned int i 
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] mu
        cdef numpy.ndarray[int, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[int, ndim=1, mode="c"] colInds
        
        rowInds, colInds = X.nonzero()

        mu = X.sum(1)/numpy.bincount(rowInds, minlength=X.shape[0])
        
        for i in range(rowInds.shape[0]): 
            X[rowInds[i], colInds[i]] = X[rowInds[i], colInds[i]] - mu[rowInds[i]] 
        
        return X
        
    @staticmethod
    def generateSparseBinaryMatrixPL(shape, unsigned int p, double density=0.2, double alpha=1, bint csarray=False):
        """
        Create an underlying matrix Z = UsV.T of rank p. Assign ratings to users and items 
        based on the power law until we have numInds indices. 
        """
        cdef unsigned int m, n, i, j
        cdef double sigma  
        cdef numpy.ndarray[double, ndim=1, mode="c"] userProbs
        cdef numpy.ndarray[double, ndim=1, mode="c"] itemProbs
        cdef numpy.ndarray[double, ndim=2, mode="c"] R
        cdef numpy.ndarray[double, ndim=2, mode="c"] U
        cdef numpy.ndarray[double, ndim=2, mode="c"] V
        cdef numpy.ndarray[double, ndim=2, mode="c"] Z
        
        m, n = shape
        p = numpy.min([n, m, p])

        R = numpy.random.rand(m, p)
        U, R = numpy.linalg.qr(R)

        R = numpy.random.rand(n, p)
        V, R = numpy.linalg.qr(R)
        
        Z = U.dot(V.T)
        sigma = numpy.mean(Z)
        
        if csarray:
            import sppy
            X = sppy.csarray(shape, storagetype="row")
        else:
            X = scipy.sparse.csr_matrix(shape)        
        
        userProbs = Util.powerLawProbs(alpha, zeroVal=0.5, maxInt=m)
        itemProbs = Util.powerLawProbs(alpha, zeroVal=0.5, maxInt=n)
        
        numpy.random.shuffle(userProbs)
        numpy.random.shuffle(itemProbs)
        
        while X.nnz/float(m*n) < density:
            #userProbs = userSums/userSums.sum()
            #i = numpy.random.choice(m, p=userProbs, replace=True)    
            i = Util.randomChoice(userProbs)
            
            #itemProbs = itemSums/itemSums.sum()
            #j = numpy.random.choice(n, p=itemProbs, replace=True)  
            j = Util.randomChoice(itemProbs)
            
            if Z[i, j] >= sigma: 
                X[i, j] = 1
                

        return X, U, V 
