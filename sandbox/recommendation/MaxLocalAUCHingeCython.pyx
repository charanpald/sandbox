#cython: profile=False 
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
from __future__ import print_function
import cython
from cython.parallel import parallel, prange
cimport numpy
import numpy
from sandbox.util.CythonUtils cimport dot, scale, choice, inverseChoice, inverseChoiceArray, uniformChoice, plusEquals, partialSum, square
from sandbox.util.SparseUtilsCython import SparseUtilsCython

"""
A simple squared hinge loss version of the objective. 
"""

from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX

cdef extern from "math.h":
    double exp(double x)
    double tanh(double x)
    bint isnan(double x)  
    double sqrt(double x)
    double fmax(double x, double y)
    
    
cdef class MaxLocalAUCHingeCython(object): 
    def __init__(self, unsigned int k=8, double lmbdaU=0.0, double lmbdaV=1.0, bint normalise=True, unsigned int numAucSamples=10, unsigned int numRowSamples=30, unsigned int startAverage=30, double rho=0.5): 
        self.itemFactors = False        
        self.k = k 
        self.lmbdaU = lmbdaU
        self.lmbdaV = lmbdaV
        self.maxNorm = 100
        self.normalise = normalise 
        self.numAucSamples = numAucSamples
        self.numRowSamples = numRowSamples
        self.printStep = 1000
        self.rho = rho
        self.startAverage = startAverage 
    
    def derivativeUi(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, unsigned int i):
        """
        Find  delta phi/delta u_i using the hinge loss.  
        """
        cdef unsigned int p, q
        cdef double uivp, uivq, gamma, kappa, ri
        cdef double  normDeltaTheta, hGamma, zeta, normGp, normGq 
        cdef unsigned int m = U.shape[0], n = V.shape[0]
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaBari 
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(self.k, numpy.float)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(self.k, numpy.float)
          
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint32), omegai, assume_unique=True)
        normGp = 0
        
        for p in omegai: 
            uivp = dot(U, i, V, p, self.k)
            normGp += gp[p]
            
            deltaBeta = numpy.zeros(self.k, numpy.float)
            kappa = 0
            zeta = 0 
            normGq = 0
            
            for q in omegaBari: 
                uivq = dot(U, i, V, q, self.k)
                
                gamma = uivp - uivq
                hGamma = max(0, 1-gamma) 
                
                normGq += gq[q]
                
                deltaBeta += (V[q, :] - V[p, :])*gq[q]*hGamma
             
            deltaTheta += deltaBeta*gp[p]/normGq
        
        if normGp != 0:
            deltaTheta /= m*normGp
        deltaTheta += scale(U, i, self.lmbdaU/m, self.k)
                    
        #Normalise gradient to have unit norm 
        normDeltaTheta = numpy.linalg.norm(deltaTheta)
        
        if normDeltaTheta != 0 and self.normalise: 
            deltaTheta = deltaTheta/normDeltaTheta
        
        return deltaTheta
        

    def objectiveApprox(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[unsigned int, ndim=1, mode="c"] allIndPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] allColInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V,  numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, bint full=False):         
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0]
        cdef unsigned int i, j, k, p, q
        cdef double uivp, uivq, gamma, kappa, ri, partialObj, hGamma, hKappa, normGp, normGq, zeta, normGpq
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] allOmegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaiSample
        cdef numpy.ndarray[double, ndim=1, mode="c"] objVector = numpy.zeros(m, dtype=numpy.float)
    
        k = U.shape[1]
        
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            allOmegai = allColInds[allIndPtr[i]:allIndPtr[i+1]]
            
            partialObj = 0
            normGp = 0                
            
            omegaiSample = uniformChoice(omegai, self.numAucSamples) 
            #omegaiSample = omegai
            
            for p in omegaiSample:
                uivp = dot(U, i, V, p, self.k)
                kappa = 0 
                normGq = 0
                normGp += gp[p]
                
                for j in range(self.numAucSamples): 
                    q = inverseChoice(allOmegai, n) 
                    uivq = dot(U, i, V, q, self.k)
                    gamma = uivp - uivq
                    hGamma = max(0, 1-gamma)
                    
                    normGq += gq[q]
                    kappa += gq[q]*square(hGamma)
                
                if normGq != 0: 
                    partialObj += gp[p]*(kappa/normGq)
               
            if normGp != 0: 
                objVector[i] = partialObj/normGp
        
        objVector /= 2*m
        objVector += (0.5/m)*((self.lmbdaV/m)*numpy.linalg.norm(V)**2 + (self.lmbdaU/m)*numpy.linalg.norm(U)**2) 
        
        if full: 
            return objVector
        else: 
            return objVector.sum()         
        
    def objective(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[unsigned int, ndim=1, mode="c"] allIndPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] allColInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, bint full=False):         
        """
        Note that distributions gp, gq and gi must be normalised to have sum 1. 
        """
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0]
        cdef unsigned int i, j, p, q
        cdef double uivp, uivq, gamma, kappa, ri, hGamma, normGp, normGq, sumQ=0
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaBari 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] allOmegai 
        cdef numpy.ndarray[double, ndim=1, mode="c"] objVector = numpy.zeros(m, dtype=numpy.float)
    
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            allOmegai = allColInds[allIndPtr[i]:allIndPtr[i+1]]
            
            omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint32), omegai, assume_unique=True)
            partialObj = 0 
            normGp = 0
            
            for p in omegai:
                uivp = dot(U, i, V, p, self.k)
                kappa = 0 
                normGq = 0
                
                normGp += gp[p]
                
                for q in omegaBari:                 
                    uivq = dot(U, i, V, q, self.k)
                    gamma = uivp - uivq
                    hGamma = max(0, 1-gamma)
                    
                    normGq += gq[q]
                    kappa += square(hGamma)*gq[q]
                
                if normGq != 0: 
                    partialObj += gp[p]*(kappa/normGq)
               
            if normGp != 0: 
                objVector[i] = partialObj/normGp
        
        objVector /= 2*m  
        objVector += (0.5/m)*((self.lmbdaV/m)*numpy.linalg.norm(V)**2 + (self.lmbdaU/m)*numpy.linalg.norm(U)**2) 
        
        if full: 
            return objVector
        else: 
            return objVector.sum() 