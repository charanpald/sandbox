# cython: profile=True 
import cython
from cython.parallel import parallel, prange
cimport numpy
import numpy
from sandbox.util.SparseUtilsCython import SparseUtilsCython

from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX

cdef extern from "math.h":
    double exp(double x)
    bint isnan(double x)    
    

@cython.nonecheck(False)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef inline int randint(int i):
    """
    Note that i must be less than RAND_MAX. 
    """
    return rand() % i   

@cython.nonecheck(False)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef inline double square(double d):
    return d*d    

@cython.nonecheck(False)
@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef inline double dot(numpy.ndarray[double, ndim = 2, mode="c"] U, unsigned int i, numpy.ndarray[double, ndim = 2, mode="c"] V, unsigned int j, unsigned int k):
    """
    Compute the dot product between U[i, :] and V[j, :]
    """
    cdef double result = 0
    cdef unsigned int s = 0
    for s in range(k):
        result += U[i, s]*V[j, s]
    return result

@cython.nonecheck(False)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef inline numpy.ndarray[double, ndim = 1, mode="c"] scale(numpy.ndarray[double, ndim = 2, mode="c"] U, unsigned int i, double d, unsigned int k):
    """
    Computes U[i, :] * d where k is U.shape[1]
    """
    cdef numpy.ndarray[double, ndim = 1, mode="c"] ui = numpy.empty(k)
    cdef unsigned int s = 0
    for s in range(k):
        ui[s] = U[i, s]*d
    return ui

@cython.nonecheck(False)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef inline numpy.ndarray[double, ndim = 1, mode="c"] plusEquals(numpy.ndarray[double, ndim = 2, mode="c"] U, unsigned int i, numpy.ndarray[double, ndim = 1, mode="c"] d, unsigned int k):
    """
    Computes U[i, :] += d[i] where k is U.shape[1]
    """
    cdef unsigned int s = 0
    for s in range(k):
        U[i, s] = U[i, s] + d[s]

@cython.nonecheck(False)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef inline numpy.ndarray[double, ndim = 1, mode="c"] plusEquals1d(numpy.ndarray[double, ndim = 1, mode="c"] u, numpy.ndarray[double, ndim = 1, mode="c"] d, unsigned int k):
    """
    Computes U[i] += d[i] 
    """
    cdef unsigned int s = 0
    for s in range(k):
        u[s] = u[s] + d[s]

@cython.nonecheck(False)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef unsigned int getNonZeroRow(X, unsigned int i, unsigned int n):
    """
    Find a random nonzero element in the ith row of X
    """
    cdef unsigned int q = numpy.random.randint(0, n)
    
    while X[i, q] != 0:
        q = numpy.random.randint(0, n)
    return q

@cython.boundscheck(False)
@cython.wraparound(False)
def derivativeUi(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int i, numpy.ndarray[double, ndim=1, mode="c"] r, double lmbda, double rho, bint normalise):
    """
    Find  delta phi/delta u_i using the hinge loss.  
    """
    cdef unsigned int p, q, ind, j, s
    cdef unsigned int k = U.shape[1]
    cdef double uivp, ri, uivq, gamma, kappa
    cdef double denom, denom2, normDeltaTheta, alpha 
    cdef unsigned int m = X.shape[0], n = X.shape[1], numOmegai, numOmegaBari
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
      
    omegai = omegaList[i]
    omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint), omegai, assume_unique=True)
    numOmegai = omegai.shape[0]
    numOmegaBari = n-numOmegai
    
    deltaTheta = numpy.zeros(k)
    ri = r[i]
    
    if numOmegai * numOmegaBari != 0:         
        for p in omegai: 
            for q in omegaBari: 
                uivp = dot(U, i, V, p, k)
                uivq = dot(U, i, V, q, k)
                
                gamma = uivp - uivq
                kappa = uivp - ri
                
                if gamma <= 1: 
                    deltaTheta += (V[q, :] - V[p, :])*(1-gamma)*(1-rho)
                
                if kappa <= 1: 
                    deltaTheta -= V[p, :]*(1-kappa)*rho
                
        deltaTheta /= float(numOmegai * numOmegaBari * m)
            
    #Add regularisation 
    deltaTheta = scale(U, i, lmbda/m, k) + deltaTheta        
        
    #Normalise gradient to have unit norm 
    normDeltaTheta = numpy.linalg.norm(deltaTheta)
    
    if normDeltaTheta != 0 and normalise: 
        deltaTheta = deltaTheta/normDeltaTheta
    
    return deltaTheta

def updateU(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, double sigma, numpy.ndarray[double, ndim=1, mode="c"] r, double nu): 
    """
    Compute the full gradient descent update of U
    """    
    
    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dU = numpy.zeros((U.shape[0], U.shape[1]), numpy.float)
    cdef unsigned int i 
    cdef unsigned int m = X.shape[0]
    cdef unsigned int k = U.shape[1]
    
    for i in range(m): 
        dU[i, :] = derivativeUi(X, U, V, omegaList, i, r, nu) 
    
    U -= sigma*dU
    
    for i in range(m):
        U[i,:] = scale(U, i, 1/numpy.linalg.norm(U[i,:]), k)   



@cython.boundscheck(False)
@cython.wraparound(False)
def derivativeUiApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int i, unsigned int numRowSamples, unsigned int numAucSamples, numpy.ndarray[double, ndim=1, mode="c"] r, double lmbda, double rho, bint normalise):
    """
    Find an approximation of delta phi/delta u_i using the simple objective without 
    sigmoid functions. 
    """
    cdef unsigned int p, q, ind, j, s
    cdef unsigned int k = U.shape[1]
    cdef double uivp, ri, uivq, gamma, kappa
    cdef double normDeltaTheta 
    cdef unsigned int m = X.shape[0], n = X.shape[1], numOmegai, numOmegaBari
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsP = numpy.zeros(k, numpy.int)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ = numpy.zeros(k, numpy.int)
         
    omegai = omegaList[i]
    omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint), omegai, assume_unique=True)
    numOmegai = omegai.shape[0]
    numOmegaBari = n-numOmegai
    
    deltaTheta = numpy.zeros(k)
    ri = r[i]
    
    if numOmegai * numOmegaBari != 0: 
        indsP = numpy.random.randint(0, numOmegai, numAucSamples)
        #indsQ = numpy.random.randint(0, numOmegaBari, numAucSamples)        
        
        for j in range(numAucSamples):
            p = omegai[indsP[j]] 
            #q = omegaBari[indsQ[j]]  
            q = getNonZeroRow(X, i, n) 
        
            uivp = dot(U, i, V, p, k)
            uivq = dot(U, i, V, q, k)
            
            gamma = uivp - uivq
            kappa = uivp - ri
            
            if gamma <= 1: 
                deltaTheta += (V[q, :] - V[p, :])*(1-gamma)*(1-rho)
            
            if kappa <= 1: 
                deltaTheta -= V[p, :]*(1-kappa)*rho
                
        deltaTheta /= float(numAucSamples * m)
            
    #Add regularisation 
    deltaTheta = scale(U, i, lmbda/m, k) + deltaTheta        
        
    #Normalise gradient to have unit norm 
    normDeltaTheta = numpy.linalg.norm(deltaTheta)
    
    if normDeltaTheta != 0 and normalise: 
        deltaTheta = deltaTheta/normDeltaTheta
    
    return deltaTheta

@cython.boundscheck(False)
@cython.wraparound(False)
def derivativeVi(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int j, numpy.ndarray[double, ndim=1, mode="c"] r, double lmbda, double rho, bint normalise): 
    """
    delta phi/delta v_i using hinge loss. 
    """
    cdef unsigned int i = 0
    cdef unsigned int k = U.shape[1]
    cdef unsigned int p, q, numOmegai, numOmegaBari, t
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1], ind
    cdef unsigned int s = 0
    cdef double uivp, uivq,  betaScale, ri, normTheta
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    
    for i in range(m): 
        omegai = omegaList[i]
        omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint), omegai, assume_unique=True)
        numOmegai = omegai.shape[0]       
        numOmegaBari = n-numOmegai
        
        ri = r[i]
        betaScale = 0
        
        if X[i, j] != 0:                 
            p = j 
            uivp = dot(U, i, V, p, k)
            kappa = uivp - ri
            
            for q in omegaBari: 
                uivq = dot(U, i, V, q, k)
                gamma = uivp - uivq
                
                
                if gamma <= 1: 
                    betaScale += (1-gamma)*(1-rho) 

            betaScale /= numOmegaBari

            if kappa <= 1: 
                betaScale += (1-kappa)*rho              
                
            deltaBeta = scale(U, i, -betaScale/numOmegai, k)
        else:
            q = j 
            uivq = dot(U, i, V, q, k)
                            
            for p in omegai: 
                uivp = dot(U, i, V, p, k)
                
                gamma = uivp - uivq               
                
                if gamma <= 1: 
                    betaScale += (1-gamma)*(1-rho)

            if numOmegai != 0:
                deltaBeta = scale(U, i, betaScale/(numOmegai*numOmegaBari), k)  
                
        deltaTheta += deltaBeta
    
    deltaTheta = deltaTheta/float(m)
        
    #Add regularisation 
    deltaTheta = scale(V, j, lmbda/m, k) + deltaTheta
    
    #Make gradient unit norm 
    normTheta = numpy.linalg.norm(deltaTheta)
    if normTheta != 0 and normalise: 
        deltaTheta = deltaTheta/normTheta
    
    return deltaTheta
 

def updateV(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, double sigma, numpy.ndarray[double, ndim=1, mode="c"] r, double lmbda): 
    """
    Compute the full gradient descent update of V
    """
    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dV = numpy.zeros((V.shape[0], V.shape[1]), numpy.float)
    cdef unsigned int i 
    cdef unsigned int n = X.shape[1]
    cdef unsigned int k = V.shape[1]
    
    for i in range(n): 
        dV[i, :] = derivativeVi(X, U, V, omegaList, i, r, lmbda) 
    
    V -= sigma*dV

    #for i in range(n):
    #    V[i,:] = scale(V, i, 1/numpy.linalg.norm(V[i,:]), k)   
       

@cython.boundscheck(False)
@cython.wraparound(False)
def derivativeViApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int j, unsigned int numRowSamples, unsigned int numAucSamples, numpy.ndarray[double, ndim=1, mode="c"] r, double lmbda, double rho, bint normalise): 
    """
    delta phi/delta v_i  using the hinge loss. 
    """
    cdef unsigned int i = 0
    cdef unsigned int k = U.shape[1]
    cdef unsigned int p, q, numOmegai, numOmegaBari, t
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1], ind
    cdef unsigned int s = 0
    cdef double uivp, uivq,  betaScale, ri, normTheta
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    #cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] rowInds = numpy.unique(numpy.array(numpy.random.randint(0, m, numRowSamples), dtype=numpy.uint))
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] rowInds = numpy.random.permutation(m)[0:numRowSamples]
    
    for i in rowInds: 
        omegai = omegaList[i]
        numOmegai = omegai.shape[0]       
        numOmegaBari = n-numOmegai
        
        ri = r[i]
        betaScale = 0
        
        if X[i, j] != 0:                 
            p = j 
            uivp = dot(U, i, V, p, k)
            kappa = uivp - ri
            #omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint), omegai, assume_unique=True)
            #omegaBari = numpy.random.permutation(omegaBari)[0:numAucSamples]
            
            for s in range(numAucSamples): 
                q = getNonZeroRow(X, i, n)
                #for q in omegaBari: 
            
                uivq = dot(U, i, V, q, k)
                gamma = uivp - uivq
                
                
                if gamma <= 1: 
                    betaScale += (1-gamma)*(1-rho) 

            betaScale /= numAucSamples

            if kappa <= 1: 
                betaScale += (1-kappa)*rho              
                
            #Note we  use numAucSamples*numOmegai to normalise
            deltaBeta = scale(U, i, -betaScale/numOmegai, k)
        else:
            q = j 
            uivq = dot(U, i, V, q, k)
                            
            for p in omegai: 
                uivp = dot(U, i, V, p, k)
                
                gamma = uivp - uivq               
                
                if gamma <= 1: 
                    betaScale += (1-gamma)*(1-rho)

            if numOmegai != 0:
                deltaBeta = scale(U, i, betaScale/(numOmegai*numOmegaBari), k)  
                
        deltaTheta += deltaBeta
    
    if rowInds.shape[0]!= 0: 
        deltaTheta = deltaTheta/float(rowInds.shape[0])
        
    #Add regularisation 
    deltaTheta = scale(V, j, lmbda/m, k) + deltaTheta
    
    #Make gradient unit norm 
    normTheta = numpy.linalg.norm(deltaTheta)
    if normTheta != 0 and normalise: 
        deltaTheta = deltaTheta/normTheta
    
    return deltaTheta


@cython.boundscheck(False)
@cython.wraparound(False)
def updateUVApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, numpy.ndarray[unsigned int, ndim=1, mode="c"] rowInds, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, unsigned int ind, double sigma, unsigned int numIterations, unsigned int numRowSamples, unsigned int numAucSamples, double w, double lmbda, double rho, bint normalise): 
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]    
    cdef unsigned int k = U.shape[1] 
    cdef unsigned int numAucSamplesR = 500
    #cdef numpy.ndarray[double, ndim=1, mode="c"] r = numpy.ones(m)*-1
    cdef numpy.ndarray[double, ndim=1, mode="c"] r = SparseUtilsCython.computeR(U, V, w, numAucSamplesR) 
    cdef numpy.ndarray[double, ndim=1, mode="c"] dUi = numpy.zeros(k)
    cdef numpy.ndarray[double, ndim=1, mode="c"] dVj = numpy.zeros(k)
    cdef unsigned int i, j, s
    
    for s in range(numIterations):
        i = rowInds[(ind + s) % m]
        dUi = derivativeUiApprox(X, U, V, omegaList, i, numRowSamples, numAucSamples, r, lmbda, rho, normalise)
        #dUi = derivativeUi(X, U, V, omegaList, i, r, nu)
        
        j = colInds[(ind + s) % n]
        dVj = derivativeViApprox(X, U, V, omegaList, j, numRowSamples, numAucSamples, r, lmbda, rho, normalise)
        #dVi = derivativeVi(X, U, V, omegaList, j, r, nu)

        plusEquals(U, i, -sigma*dUi, k)
        
        normUi = numpy.linalg.norm(U[i,:])
        if normUi != 0: 
            U[i,:] = scale(U, i, 1/normUi, k)             
        
        plusEquals(V, j, -sigma*dVj, k)  
        
        #normVj = numpy.linalg.norm(V[j,:])
        #if normVj > 1: 
        #    V[j,:] = scale(V, j, 1/normVj, k)  
        
@cython.boundscheck(False)
@cython.wraparound(False)   
def objectiveApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int numAucSamples, numpy.ndarray[double, ndim=1, mode="c"] r, double lmbda, double rho):         
    cdef double obj = 0 
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]
    cdef unsigned int i, j, k, p, q
    cdef double kappa, uivp, uivq, gamma, partialObj
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(10, numpy.uint)  
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsP
    
    k = U.shape[1]
    
    for i in range(m): 
        omegai = omegaList[i]
        #omegaBari = numpy.setdiff1d(numpy.arange(n), omegai, assume_unique=True)
        
        ri = r[i]
        
        if omegai.shape[0] * (n-omegai.shape[0]) != 0: 
            partialObj = 0                
            
            indsP = numpy.random.randint(0, omegai.shape[0], numAucSamples)  
            #indsQ = numpy.random.randint(0, omegaBari.shape[0], numAucSamples)
            
            for j in range(numAucSamples):
                p = omegai[indsP[j]] 
                #q = omegaBari[indsQ[j]]
                q = getNonZeroRow(X, i, n)                  
            
                uivp = dot(U, i, V, p, k)
                gamma = uivp - uivq
                
                uivq = dot(U, i, V, q, k)
                kappa = uivp - ri
                
                if gamma <= 1: 
                    partialObj += ((1-gamma)**2) * (1-rho)
                    
                if kappa <= 1: 
                    partialObj += ((1-kappa)**2) * rho
                        
            obj += partialObj/float(numAucSamples)
    
    obj /= 2*m       
    obj += (lmbda/(2*m))*numpy.linalg.norm(V)**2 
    
    return obj 
  
@cython.boundscheck(False)
@cython.wraparound(False)  
def localAUCApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int numAucSamples, numpy.ndarray[double, ndim=1, mode="c"] r): 
    """
    Compute the estimated local AUC for the score functions UV^T relative to X with 
    quantile vector r. If evaluating on a set of test observations then X is 
    trainX+testX and omegaList is from testX. 
    """
    
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(10, numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] localAucArr = numpy.zeros(m)
    cdef unsigned int i, j, k, ind, p, q, nOmegai
    cdef double partialAuc, ri, uivp

    k = U.shape[1]

    for i in range(m): 
        omegai = omegaList[i]
        nOmegai = omegai.shape[0]
        #omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
        ri = r[i]
        
        if nOmegai * (n-nOmegai) != 0: 
            partialAuc = 0                
            
            for j in range(numAucSamples):
                #ind = numpy.random.randint(omegai.shape[0])
                ind = randint(nOmegai)
                p = omegai[ind] 
                
                q = getNonZeroRow(X, i, n)                
                uivp = dot(U, i, V, p, k)

                if uivp > ri and uivp > dot(U, i, V, q, k): 
                    partialAuc += 1 
                        
            localAucArr[i] = partialAuc/float(numAucSamples)     
    
    return localAucArr.mean()