# cython: profile=True 
import cython
#from math import exp 
cimport numpy
import numpy

cdef extern from "math.h":
    double exp(double x)

@cython.boundscheck(False)
def derivativeUiApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, omegaList, unsigned int i, unsigned int mStar, unsigned int sampleSize, unsigned int k, double lmbda, numpy.ndarray[double, ndim=1, mode="c"] r): 
    """
    delta phi/delta u_i
    """
    cdef unsigned int p, q, ind, j
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
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsP = numpy.zeros(k, numpy.int)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ = numpy.zeros(k, numpy.int)
    
    deltaPhi = lmbda * U[i, :]
    
    omegai = omegaList[i]
    omegaBari = numpy.setdiff1d(numpy.arange(X.shape[1], dtype=numpy.uint), omegai, assume_unique=True)
    
    deltaAlpha = numpy.zeros(k)
    
    ui = U[i, :]
    ri = r[i]
    uiV = ui.dot(V.T)
    
    if omegai.shape[0] * omegaBari.shape[0] != 0: 
        indsP = numpy.random.randint(0, omegai.shape[0], sampleSize)
        indsQ = numpy.random.randint(0, omegaBari.shape[0], sampleSize)        
        
        for j in range(sampleSize):
            #Maybe sample without replacement 
            p = omegai[indsP[j]] 
            q = omegaBari[indsQ[j]]  
        
            vp = V[p, :]
            uivp = uiV[p]
            kappa = exp(-uivp +ri)
            onePlusKappa = 1+kappa
            onePlusKappSq = onePlusKappa**2
            
            vq = V[q, :]
            uivq = uiV[q]
            gamma = exp(-uivp+uivq)
            onePlusGamma = 1+gamma
            
            denom = onePlusGamma**2 * onePlusKappSq
            deltaAlpha += vp*(gamma*onePlusKappa/denom) + vq*((kappa-gamma)/denom) 
                
        
        deltaAlpha /= float(sampleSize*mStar)
            
        deltaPhi -= deltaAlpha
    
    return deltaPhi

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
cdef inline double updateDeltaBeta1(double uivq, double uivp, double onePlusKappaSq, double kappa, double twoKappa):
    gamma = exp(-uivp+uivq)
    onePlusGamma = 1+gamma
    
    denom = onePlusGamma**2 * onePlusKappaSq
    return ((gamma+kappa+gamma*twoKappa)/denom)

@cython.boundscheck(False)
def derivativeViApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, omegaList, unsigned int j, unsigned int sampleSize, unsigned int k, double lmbda, numpy.ndarray[double, ndim=1, mode="c"] r): 
    """
    delta phi/delta v_j
    """
    cdef unsigned int mStar = 0
    cdef unsigned int i = 0
    cdef unsigned int p, q, numOmegai, numOmegaBari, indP, indQ
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]
    cdef unsigned int s = 0
    cdef double uivp, kappa, onePlusKappa, uivq, gamma, onePlusGamma, denom, riExp, uivpExp
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaPhi = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaAlpha = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] ui = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] vp = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] vq = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] allInds = numpy.arange(n, dtype=numpy.uint)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsI = numpy.zeros(k, numpy.int)
    
    deltaPhi = lmbda * V[j, :]
    indsI = numpy.random.randint(0, m, sampleSize)
    #precompute U[indsI, :].V.T

    for s in range(sampleSize): 
        i = indsI[s]
        omegai = omegaList[i]
        numOmegai = omegai.shape[0]       
        numOmegaBari = n-numOmegai
        
        indP = numpy.random.randint(0, omegai.shape[0])
        
        deltaBeta = numpy.zeros(k) 
        ui = U[i, :]
        riExp = exp(r[i])
        
        if X[i, j] != 0: 
            #if j not in omega:
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            indQ = numpy.random.randint(0, omegaBari.shape[0])
            
            p = j 
            uivp = ui.T.dot(V[p, :])
            #uivp = UV[s, p]
            uivpExp = exp(uivp)            
            
            kappa = riExp/uivpExp
            onePlusKappaSq = (1+kappa)**2
            
            #Pick a random negative label
            q = omegaBari[indQ] 
            uivq = ui.T.dot(V[q, :])
            gamma = exp(uivq)/uivpExp
            
            denom = (1+gamma)**2 * onePlusKappaSq 
            deltaBeta += ui*((gamma+kappa+ 2*gamma*kappa)/denom)
            deltaAlpha += deltaBeta/float(numOmegai)
        else:
            q = j 
            uivq = ui.T.dot(V[q, :])
            
            p = omegai[indP] 
            uivp = ui.T.dot(V[p, :])
            uivpExp = exp(uivp)
            
            gamma = exp(uivq)/uivpExp
            kappa = riExp/uivpExp
            
            deltaBeta -= ui* (gamma/((1+gamma)**2 * (1+kappa)))
            deltaAlpha += deltaBeta/float(numOmegaBari)
    
    deltaAlpha /= float(sampleSize)
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
            mStar += 1
            
        deltaAlpha += deltaBeta 
    
    deltaAlpha /= float(mStar)
    deltaPhi -= deltaAlpha
    
    return deltaPhi
   
@cython.profile(False)
cdef inline double square(double d):
    return d*d    
   
@cython.boundscheck(False)
def derivativeVApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, omegaList, numpy.ndarray[long, ndim=1, mode="c"] indsJ, unsigned int sampleSize, unsigned int k, double lmbda, numpy.ndarray[double, ndim=1, mode="c"] r): 
    """
    delta phi/delta V using a few randomly selected rows of V
    """
    cdef unsigned int mStar = 0
    cdef unsigned int i = 0
    cdef unsigned int p, q, numOmegai, numOmegaBari, indP, indQ, t
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]
    cdef unsigned int s = 0
    cdef double uivp, kappa, onePlusKappa, uivq, gamma, onePlusGamma, denom, riExp, uivpExp
    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dV = numpy.zeros((indsJ.shape[0], k), numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaAlpha = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] ui = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] allInds = numpy.arange(n, dtype=numpy.uint)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsI = numpy.zeros(k, numpy.int)
    
    dV = lmbda * V[indsJ, :]
    
    for t in xrange(indsJ.shape[0]):
        j = indsJ[t]
        
        indsI = numpy.random.randint(0, m, sampleSize)
    
        for s in xrange(sampleSize): 
            i = indsI[s]
            omegai = omegaList[i]
            numOmegai = omegai.shape[0]       
            
            indP = numpy.random.randint(0, omegai.shape[0])
            
            ui = U[i, :]
            riExp = exp(r[i])
            
            if X[i, j] != 0: 
                #if j not in omega:
                q = numpy.random.randint(0, n)
                while X[i, q] != 0: 
                    q = numpy.random.randint(0, n)
                
                p = j 
                
                uivp = ui.T.dot(V[p, :])
                #uivp = dot(ui, V[p, :])
                uivpExp = exp(uivp)            
                kappa = riExp/uivpExp

                uivq = ui.T.dot(V[q, :])
                #uivq = dot(ui, V[q, :])
                gamma = exp(uivq)/uivpExp
                
                denom = square(1+gamma) * square(1+kappa) 
                deltaBeta = ui*((gamma+kappa+ 2*gamma*kappa)/(denom*numOmegai))
                deltaAlpha += deltaBeta
            else:
                q = j 
                uivq = ui.T.dot(V[q, :])
                #uivq = dot(ui, V[q, :])
                
                p = omegai[indP] 
                uivp = ui.T.dot(V[p, :])
                #uivp = dot(ui, V[p, :])
                uivpExp = exp(uivp)
                
                gamma = exp(uivq)/uivpExp
                kappa = riExp/uivpExp
                
                numOmegaBari = n-numOmegai
                denom = square(1+gamma) * (1+kappa) 
                deltaBeta = ui* (gamma/(denom * float(numOmegaBari)))
                deltaAlpha -= deltaBeta
        
        deltaAlpha /= float(sampleSize)
        dV[t, :] -= deltaAlpha
    
    return dV