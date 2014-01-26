# cython: profile=True 
import cython
#from math import exp 
cimport numpy
import numpy

cdef extern from "math.h":
    double exp(double x)

@cython.profile(False)
cdef inline double square(double d):
    return d*d    

@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef inline double dot(numpy.ndarray[double, ndim = 2, mode="c"] U, unsigned int i, numpy.ndarray[double, ndim = 2, mode="c"] V, unsigned int j, unsigned int k):
    cdef double result = 0
    cdef unsigned int s = 0
    for s in range(k):
        result += U[i, s]*V[j, s]
    return result

@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef inline numpy.ndarray[double, ndim = 1, mode="c"] scale(numpy.ndarray[double, ndim = 2, mode="c"] U, unsigned int i, double d, unsigned int k):
    cdef numpy.ndarray[double, ndim = 1, mode="c"] ui = numpy.empty(k)
    cdef unsigned int s = 0
    for s in range(k):
        ui[s] = U[i, s]*d
    return ui



@cython.boundscheck(False)
def derivativeUi(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int i, unsigned int k, double lmbda, numpy.ndarray[double, ndim=1, mode="c"] r): 
    """
    delta phi/delta u_i
    """
    cdef unsigned int p, q, m 
    cdef double uivp, ri, uivq, kappa, onePlusKappa, onePlusKappaSq, gamma, onePlusGamma
    cdef double denom, denom2
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaPhi = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaAlpha = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] ui = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] vp = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] vq = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] uiV = numpy.zeros(k, numpy.float)
    
    m = X.shape[0]
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
        onePlusKappaSq = square(onePlusKappa)
        
        for q in omegaBari: 
            vq = V[q, :]
            uivq = uiV[q]
            gamma = exp(-uivp+uivq)
            onePlusGamma = 1+gamma
            onePlusGammaSq = square(onePlusGamma)
            
            denom = onePlusGammaSq * onePlusKappaSq
            denom2 = onePlusGammaSq * onePlusKappa
            deltaAlpha += vp*((gamma+kappa+2*gamma*kappa)/denom) - vq*(gamma/denom2) 
            
    if omegai.shape[0] * omegaBari.shape[0] != 0: 
        deltaAlpha /= float(omegai.shape[0] * omegaBari.shape[0]*m)
        
    deltaPhi -= deltaAlpha
    
    return deltaPhi


def updateUApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int numAucSamples, double sigma, unsigned int numIterations, unsigned int k, double lmbda, numpy.ndarray[double, ndim=1, mode="c"] r):
    """
    delta phi/delta U
    """
    cdef unsigned int p, q, ind, j, s
    cdef double uivp, ri, uivq, kappa, onePlusKappa, onePlusKappaSq, gamma, onePlusGamma
    cdef double denom, denom2
    cdef unsigned int n, m, numOmegai, numOmegaBari
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaPhi = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaAlpha = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsP = numpy.zeros(k, numpy.int)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ = numpy.zeros(k, numpy.int)

    m = X.shape[0]
    n = X.shape[1]
    
    for s in range(numIterations): 
        i = numpy.random.randint(m)  
        deltaPhi = U[i, :]*lmbda    
         
        omegai = omegaList[i]
        omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint), omegai, assume_unique=True)
        numOmegai = omegai.shape[0]
        numOmegaBari = omegaBari.shape[0]
        
        deltaAlpha = numpy.zeros(k)
        
        ri = r[i]
        
        if omegai.shape[0] * omegaBari.shape[0] != 0: 
            indsP = numpy.random.randint(0, numOmegai, numAucSamples)
            indsQ = numpy.random.randint(0, numOmegaBari, numAucSamples)        
            
            for j in range(numAucSamples):
                p = omegai[indsP[j]] 
                q = omegaBari[indsQ[j]]  
            
                uivp = dot(U, i, V, p, k)
                kappa = exp(-uivp +ri)
                onePlusKappa = 1+kappa
                onePlusKappaSq = square(onePlusKappa)
                
                uivq = dot(U, i, V, q, k)
                gamma = exp(-uivp+uivq)
                onePlusGamma = 1+gamma
                onePlusGammaSq = square(onePlusGamma)
                
                denom = onePlusGammaSq * onePlusKappaSq
                denom2 = onePlusGammaSq * onePlusKappa
                deltaAlpha += scale(V, p, ((gamma+kappa+2*gamma*kappa)/denom), k) - scale(V, q, (gamma/denom2), k) 
                    
            deltaAlpha /= float(numAucSamples)
            deltaPhi -= deltaAlpha
    
        #Could do this in Cython 
        U[i,:] -= deltaPhi*sigma 


@cython.boundscheck(False)
def derivativeVi(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int j, unsigned int k, double lmbda, numpy.ndarray[double, ndim=1, mode="c"] r): 
    """
    delta phi/delta v_j
    """
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
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] uiVexp = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] riExp = numpy.zeros(k, numpy.float)
    
    deltaPhi = lmbda * V[j, :]

    riExp = numpy.exp(r)

    for i in range(m): 
        omegai = omegaList[i]
        
        deltaBeta = numpy.zeros(k) 
        ui = U[i, :]
        ri = r[i]
        uiV = ui.dot(V.T)
        uiVexp = numpy.exp(-uiV)
        
        if X[i, j] != 0: 
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            p = j 
            uivp = uiV[p]
            kappa = uiVexp[p]*riExp[i]
            onePlusKappa = 1+kappa
            onePlusKappaSq = onePlusKappa**2
            twoKappa = 2*kappa
            
            for q in omegaBari: 
                uivq = uiV[q]
                gamma = uiVexp[p]/uiVexp[q]
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
                
                deltaBeta -= ui* (gamma/((1+gamma)**2 * (1+kappa)))
        
        numOmegai = omegai.shape[0]       
        numOmegaBari = n-numOmegai
        
        if numOmegai*numOmegaBari != 0: 
            deltaBeta /= float(numOmegai*numOmegaBari)
            
        deltaAlpha += deltaBeta 
    
    deltaAlpha /= float(m)
    deltaPhi -= deltaAlpha
    
    return deltaPhi
   
@cython.boundscheck(False)
@cython.wraparound(False)
def updateVApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int numRowSamples, unsigned int numAucSamples, double sigma, unsigned int numIterations, unsigned int k, double lmbda, numpy.ndarray[double, ndim=1, mode="c"] r): 
    """
    delta phi/delta V using a few randomly selected rows of V
    """
    cdef unsigned int i = 0
    cdef unsigned int p, q, numOmegai, numOmegaBari, indP, indQ, t
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1], ind
    cdef unsigned int s = 0
    cdef double uivp, kappa, onePlusKappa, uivq, gamma, onePlusGamma, denom, riExp, uivpExp
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] dV = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaAlpha = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsI = numpy.zeros(numRowSamples, numpy.int)
    
    for t in xrange(numIterations):
        j = numpy.random.randint(n)
        dV = lmbda * V[j, :]
        indsI = numpy.random.randint(0, m, numRowSamples)
    
        for s in xrange(numRowSamples): 
            i = indsI[s]
            omegai = omegaList[i]
            numOmegai = omegai.shape[0]     
            riExp = exp(r[i])
            
            for ind in range(numAucSamples): 
                
                if X[i, j] != 0: 
                    #if j in omega:
                    q = numpy.random.randint(0, n)
                    while X[i, q] != 0: 
                        q = numpy.random.randint(0, n)
                    
                    p = j 
                    
                    uivp = dot(U, i, V, p, k)
                    uivpExp = exp(uivp)            
                    kappa = riExp/uivpExp
    
                    uivq = dot(U, i, V, q, k)
                    gamma = exp(uivq)/uivpExp
                    
                    denom = square(1+gamma) * square(1+kappa) 
                    deltaBeta = scale(U, i, ((gamma+kappa+ 2*gamma*kappa)/(denom*numAucSamples)), k)
                    deltaAlpha += deltaBeta
                else:
                    indP = numpy.random.randint(0, omegai.shape[0])                    
                    
                    q = j 
                    uivq = dot(U, i, V, q, k)
                    
                    p = omegai[indP] 
                    uivp = dot(U, i, V, p, k)
                    uivpExp = exp(uivp)
                    
                    gamma = exp(uivq)/uivpExp
                    kappa = riExp/uivpExp
                    
                    numOmegaBari = n-numOmegai
                    denom = square(1+gamma) * (1+kappa) 
                    deltaBeta = scale(U, i, gamma/(denom * numAucSamples), k)
                    deltaAlpha -= deltaBeta
        
        deltaAlpha /= float(numRowSamples)
        dV -= deltaAlpha
        
        V[j, :] -= sigma*dV
    

    
def objectiveApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int numAucSamples, double lmbda, numpy.ndarray[double, ndim=1, mode="c"] r):         
    cdef double obj = 0 
    cdef unsigned int m = X.shape[0]
    cdef unsigned int i, j, k, p, q
    cdef double kappa, onePlusKappa, uivp, uivq, gamma, partialAuc
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] allInds = numpy.arange(X.shape[1])   
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(10, numpy.uint)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] omegaBari    
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsP
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ
    
    k = U.shape[1]
    
    for i in range(X.shape[0]): 
        omegai = omegaList[i]
        omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
        
        ri = r[i]
        
        if omegai.shape[0] * omegaBari.shape[0] != 0: 
            partialAuc = 0                
            
            indsP = numpy.random.randint(0, omegai.shape[0], numAucSamples)  
            indsQ = numpy.random.randint(0, omegaBari.shape[0], numAucSamples)
            
            for j in range(numAucSamples):
                p = omegai[indsP[j]] 
                q = omegaBari[indsQ[j]]                  
            
                uivp = dot(U, i, V, p, k)
                kappa = exp(-uivp+ri)
                
                uivq = dot(U, i, V, q, k)
                gamma = exp(-uivp+uivq)

                partialAuc += 1/((1+gamma) * (1+kappa))
                        
            obj += partialAuc/float(numAucSamples)
    
    obj /= m       
    obj = 0.5*lmbda * (numpy.sum(U**2) + numpy.sum(V**2)) - obj
    
    return obj 
    
def localAUCApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int numAucSamples, numpy.ndarray[double, ndim=1, mode="c"] r): 
    """
    Compute the estimated local AUC for the score functions UV^T relative to X with 
    quantile vector r. 
    """
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] allInds = numpy.arange(X.shape[1]) 
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(10, numpy.uint)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] omegaBari 
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] localAucArr = numpy.zeros(X.shape[0])
    cdef unsigned int i, j, k, ind, p, q
    cdef double partialAuc
    
    k = U.shape[1]

    for i in range(X.shape[0]): 
        omegai = omegaList[i]
        omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
        
        if omegai.shape[0] * omegaBari.shape[0] != 0: 
            partialAuc = 0                
            
            for j in range(numAucSamples):
                ind = numpy.random.randint(omegai.shape[0])
                p = omegai[ind] 
                
                ind = numpy.random.randint(omegaBari.shape[0])
                q = omegaBari[ind]   
                
                if dot(U, i, V, p, k) > dot(U, i, V, q, k) and dot(U, i, V, p, k) > r[i]: 
                    partialAuc += 1 
                        
            localAucArr[i] = partialAuc/float(numAucSamples)     
    
    return localAucArr.mean()