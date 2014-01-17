# cython: profile=True 
import cython
#from math import exp 
cimport numpy
import numpy

cdef extern from "math.h":
    double exp(double x)

@cython.boundscheck(False)
def derivativeUi(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, omegaList, unsigned int i, unsigned int mStar, unsigned int k, double lmbda, numpy.ndarray[double, ndim=1, mode="c"] r): 
    """
    delta phi/delta u_i
    """
    cdef unsigned int p, q
    cdef double uivp, ri, uivq, kappa, onePlusKappa, onePlusKappSq, gamma, onePlusGamma
    cdef double denom
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaPhi = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaAlpha = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] ui = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] vp = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] vq = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] uiV = numpy.zeros(k, numpy.float)
    
    deltaPhi = lmbda * U[i, :]
    
    omegai = omegaList[i]
    omegaBari = numpy.setdiff1d(numpy.arange(X.shape[1], dtype=numpy.uint), omegai, assume_unique=True)
    
    deltaAlpha = numpy.zeros(k)
    
    ui = U[i, :]
    ri = r[i]
    uiV = ui.dot(V.T)
    
    for p in omegai: 
        vp = V[p, :]
        uivp = uiV[p]
        kappa = exp(-uivp +ri)
        onePlusKappa = 1+kappa
        onePlusKappSq = onePlusKappa**2
        
        for q in omegaBari: 
            vq = V[q, :]
            uivq = uiV[q]
            gamma = exp(-uivp+uivq)
            onePlusGamma = 1+gamma
            
            denom = onePlusGamma**2 * onePlusKappSq
            deltaAlpha += vp*(gamma*onePlusKappa/denom) + vq*((kappa-gamma)/denom) 
            
    if omegai.shape[0] * omegaBari.shape[0] != 0: 
        deltaAlpha /= float(omegai.shape[0] * omegaBari.shape[0]*mStar)
        
    deltaPhi -= deltaAlpha
    
    return deltaPhi


@cython.boundscheck(False)
def derivativeVi(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, omegaList, unsigned int j, unsigned int k, double lmbda, numpy.ndarray[double, ndim=1, mode="c"] r): 
    """
    delta phi/delta v_j
    """
    cdef unsigned int mStar = 0
    cdef unsigned int i = 0
    cdef unsigned int p, q, numOmegai, numOmegaBari
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]
    cdef double uivp, kappa, onePlusKappa, uivq, gamma, onePlusGamma, denom, ri
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaPhi = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaAlpha = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] ui = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] vp = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] vq = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] allInds = numpy.arange(n, dtype=numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] uiV = numpy.zeros(k, numpy.float)
    
    deltaPhi = lmbda * V[j, :]

    for i in range(m): 
        omegai = omegaList[i]
        
        deltaBeta = numpy.zeros(k) 
        ui = U[i, :]
        ri = r[i]
        uiV = ui.dot(V.T)
        
        if X[i, j] != 0: 
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            p = j 
            uivp = uiV[p]
            kappa = exp(-uivp + ri)
            onePlusKappa = 1+kappa
            onePlusKappaSq = onePlusKappa**2
            twoKappa = 2*kappa
            
            for q in omegaBari: 
                uivq = uiV[q]
                gamma = exp(-uivp+uivq)
                onePlusGamma = 1+gamma
                
                denom = onePlusGamma**2 * onePlusKappaSq
                deltaBeta += ui*((gamma+kappa+gamma*twoKappa)/denom)
        else:
            q = j 
            uivq = uiV[q]
            
            for p in omegai: 
                uivp = uiV[p]
                
                gamma = exp(-uivp+uivq)
                kappa = exp(-uivp+ri)
                
                deltaBeta += -ui* (gamma/((1+gamma)**2 * (1+kappa)))
        
        numOmegai = omegai.shape[0]       
        numOmegaBari = n-numOmegai
        
        if numOmegai*numOmegaBari != 0: 
            deltaBeta /= float(numOmegai*numOmegaBari)
            mStar += 1
            
        deltaAlpha += deltaBeta 
    
    deltaAlpha /= float(mStar)
    deltaPhi -= deltaAlpha
    
    return deltaPhi