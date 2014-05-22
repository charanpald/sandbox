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
    double sqrt(double x)
    

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
    cdef double e1, e2
    for s in range(k):
        e1 = U[i, s]
        e2 = V[j, s]
        result += e1*e2
    return result


@cython.nonecheck(False)
@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef inline double normRow(numpy.ndarray[double, ndim = 2, mode="c"] U, unsigned int i, unsigned int k):
    """
    Compute the dot product between U[i, :] and V[j, :]
    """
    cdef double result = 0
    cdef unsigned int s = 0
    for s in range(k):
        result += square(U[i, s])

    return sqrt(result)

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
def derivativeUi(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int i, numpy.ndarray[double, ndim=1, mode="c"] xi, double lmbda, double C, bint normalise):
    """
    Find  delta phi/delta u_i using the hinge loss.  
    """
    cdef unsigned int p, q, ind, j, s
    cdef unsigned int k = U.shape[1]
    cdef double uivp, ri, uivq, gamma, kappa
    cdef double denom, denom2, normDeltaTheta, alpha, zeta 
    cdef unsigned int m = X.shape[0], n = X.shape[1], numOmegai, numOmegaBari
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
      
    omegai = omegaList[i]
    omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint), omegai, assume_unique=True)
    numOmegai = omegai.shape[0]
    numOmegaBari = n-numOmegai
    
    deltaTheta = numpy.zeros(k)
    
    if numOmegai * numOmegaBari != 0:         
        for p in omegai: 
            vpScale = 0
            
            for q in omegaBari: 
                uivp = dot(U, i, V, p, k)
                uivq = dot(U, i, V, q, k)
                
                gamma = uivp - uivq
                zeta = 1 - gamma - xi[i]
                                
                if zeta > 0: 
                    vpScale -= zeta
                    deltaTheta += V[q, :]*zeta
                
            deltaTheta += V[p, :]*vpScale 
                
        deltaTheta /= float(numOmegai * numOmegaBari * m)
            
    #Add regularisation 
    #deltaTheta = scale(U, i, lmbda/m, k) + deltaTheta        
        
    #Normalise gradient to have unit norm 
    normDeltaTheta = numpy.linalg.norm(deltaTheta)
    
    if normDeltaTheta != 0 and normalise: 
        deltaTheta = deltaTheta/normDeltaTheta
    
    return deltaTheta

def updateU(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, double sigma, numpy.ndarray[double, ndim=1, mode="c"] r, double lmbda, double rho, bint normalise): 
    """
    Compute the full gradient descent update of U
    """    
    
    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dU = numpy.zeros((U.shape[0], U.shape[1]), numpy.float)
    cdef unsigned int i 
    cdef unsigned int m = X.shape[0]
    cdef unsigned int k = U.shape[1]
    
    for i in range(m): 
        dU[i, :] = derivativeUi(X, U, V, omegaList, i, r, lmbda, rho, normalise) 
    
    U -= sigma*dU
    
    for i in range(m):
        U[i,:] = scale(U, i, 1/numpy.linalg.norm(U[i,:]), k)   



@cython.boundscheck(False)
@cython.wraparound(False)
def derivativeUiApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, list omegaProbabilitiesList, unsigned int i, unsigned int numRowSamples, unsigned int numAucSamples, numpy.ndarray[double, ndim=1, mode="c"] xi, double lmbda, double C, bint normalise):
    """
    Find an approximation of delta phi/delta u_i using the simple objective without 
    sigmoid functions. 
    """
    cdef unsigned int p, q, ind, j, s
    cdef unsigned int k = U.shape[1]
    cdef double uivp, ri, uivq, gamma, kappa
    cdef double normDeltaTheta, vqScale, vpScale, zeta  
    cdef unsigned int m = X.shape[0], n = X.shape[1], numOmegai, numOmegaBari
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] omegaProbsi = numpy.zeros(k)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaiSample = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ = numpy.zeros(k, numpy.int)
         
    omegai = omegaList[i]
    omegaProbsi = omegaProbabilitiesList[i]
    omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint), omegai, assume_unique=True)
    numOmegai = omegai.shape[0]
    numOmegaBari = n-numOmegai
    
    deltaTheta = numpy.zeros(k)
    
    if numOmegai * numOmegaBari != 0: 
        #omegaiSample = numpy.random.choice(omegai, numAucSamples, p=omegaProbsi)
        omegaiSample = omegai
        #indsQ = numpy.random.randint(0, numOmegaBari, numAucSamples)        
        
        for p in omegaiSample:
            #p = omegai[indsP[j]] 
            q = getNonZeroRow(X, i, n) 
        
            uivp = dot(U, i, V, p, k)
            uivq = dot(U, i, V, q, k)
            
            gamma = uivp - uivq
            zeta = 1 - gamma - xi[i]
            
            if zeta > 0:             
                deltaTheta += V[q, :]*zeta - V[p, :]*zeta 
            
        deltaTheta /= float(omegaiSample.shape[0] * m)
            
    #Add regularisation 
    #deltaTheta = scale(U, i, lmbda/m, k) + deltaTheta        
        
    #Normalise gradient to have unit norm 
    normDeltaTheta = numpy.linalg.norm(deltaTheta)
    
    if normDeltaTheta != 0 and normalise: 
        deltaTheta = deltaTheta/normDeltaTheta
    
    return deltaTheta

@cython.boundscheck(False)
@cython.wraparound(False)
def derivativeVi(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int j, numpy.ndarray[double, ndim=1, mode="c"] xi, double lmbda, double C, bint normalise): 
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
        
        betaScale = 0
        
        if X[i, j] != 0:                 
            p = j 
            uivp = dot(U, i, V, p, k)
            
            for q in omegaBari: 
                uivq = dot(U, i, V, q, k)
                gamma = uivp - uivq
                zeta = 1 - gamma - xi[i]

                if zeta > 0: 
                    betaScale += zeta

            deltaBeta = scale(U, i, -betaScale/(numOmegai*numOmegaBari), k)
        else:
            q = j 
            uivq = dot(U, i, V, q, k)
                            
            for p in omegai: 
                uivp = dot(U, i, V, p, k)
                gamma = uivp - uivq  
                zeta = 1 - gamma - xi[i]
                
                if zeta > 0:  
                    betaScale += zeta

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
def derivativeViApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, list omegaProbabilitiesList, unsigned int j, unsigned int numRowSamples, unsigned int numAucSamples, numpy.ndarray[double, ndim=1, mode="c"] xi, double lmbda, double C, bint normalise): 
    """
    delta phi/delta v_i  using the hinge loss. 
    """
    cdef unsigned int i = 0
    cdef unsigned int k = U.shape[1]
    cdef unsigned int p, q, numOmegai, numOmegaBari
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]
    cdef unsigned int s = 0
    cdef double uivp, uivq,  betaScale, xii, normTheta, zeta, gamma, nu
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] omegaProbsi = numpy.zeros(k)
    #cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] rowInds = numpy.unique(numpy.array(numpy.random.randint(0, m, numRowSamples), dtype=numpy.uint))
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] rowInds = numpy.random.permutation(m)[0:numRowSamples]
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaiSample 
    
    for i in rowInds: 
        omegai = omegaList[i]
        omegaProbsi = omegaProbabilitiesList[i]
        numOmegai = omegai.shape[0]       
        numOmegaBari = n-numOmegai
        
        betaScale = 0
        xii = xi[i]
        
        if X[i, j] != 0:                 
            p = j 
            uivp = dot(U, i, V, p, k)
            nu = 1 - uivp - xii

            for s in range(numAucSamples): 
                q = getNonZeroRow(X, i, n)
                uivq = dot(U, i, V, q, k)
                #gamma = uivp - uivq
                
                zeta = nu + uivq 
                                
                if zeta > 0: 
                    betaScale += zeta
                
            deltaBeta = scale(U, i, -betaScale/(numOmegai*numAucSamples), k)
        else:
            q = j 
            uivq = dot(U, i, V, q, k)
            nu = 1 + uivq - xii

            omegaiSample = numpy.random.choice(omegai, numAucSamples, p=omegaProbsi)
            #omegaiSample = omegai
            
            for p in omegaiSample: 
                uivp = dot(U, i, V, p, k)
                #gamma = uivp - uivq
                zeta = nu - uivp
                if zeta > 0: 
                    betaScale += zeta

            if numOmegai != 0:
                deltaBeta = scale(U, i, betaScale/(omegaiSample.shape[0]*numOmegaBari), k)  
                
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
def derivativeXi(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int i, numpy.ndarray[double, ndim=1, mode="c"] xi, double lmbda, double C, bint normalise):
    """
    Find  delta phi/delta u_i using the hinge loss.  
    """
    cdef unsigned int p, q, ind, j, s
    cdef unsigned int k = U.shape[1]
    cdef double uivp, ri, uivq, gamma, kappa
    cdef double denom, denom2, normDeltaTheta, alpha, zeta, deltaTheta
    cdef unsigned int m = X.shape[0], n = X.shape[1], numOmegai, numOmegaBari
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaBari = numpy.zeros(k, numpy.uint)
      
    omegai = omegaList[i]
    omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint), omegai, assume_unique=True)
    numOmegai = omegai.shape[0]
    numOmegaBari = n-numOmegai
    
    deltaTheta = 0
    
    if numOmegai * numOmegaBari != 0:         
        for p in omegai:             
            for q in omegaBari: 
                uivp = dot(U, i, V, p, k)
                uivq = dot(U, i, V, q, k)
                
                gamma = uivp - uivq
                zeta = 1 - gamma - xi[i]
                                
                if zeta > 0: 
                    deltaTheta += zeta
                
        deltaTheta /= float(numOmegai * numOmegaBari * m)
    
    return deltaTheta


@cython.boundscheck(False)
@cython.wraparound(False)
def derivativeXiiApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, list omegaProbabilitiesList, unsigned int i, unsigned int numRowSamples, unsigned int numAucSamples, numpy.ndarray[double, ndim=1, mode="c"] xi, double lmbda, double C, bint normalise):
    """
    Find an approximation of delta phi/delta u_i using the simple objective without 
    sigmoid functions. 
    """
    cdef unsigned int p, q, ind, j, s
    cdef unsigned int k = U.shape[1]
    cdef double uivp, ri, uivq, gamma, kappa
    cdef double normDeltaTheta, vqScale, vpScale, zeta 
    cdef unsigned int m = X.shape[0], n = X.shape[1], numOmegai, numOmegaBari
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] omegaProbsi = numpy.zeros(k)
    cdef double deltaTheta = 0
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaiSample = numpy.zeros(k, numpy.uint)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ = numpy.zeros(k, numpy.int)
                
    omegai = omegaList[i]
    omegaProbsi = omegaProbabilitiesList[i]
    numOmegai = omegai.shape[0]
    numOmegaBari = n-numOmegai
    
    deltaTheta = 0
    
    if numOmegai * numOmegaBari != 0: 
        #omegaiSample = numpy.random.choice(omegai, numAucSamples, p=omegaProbsi)
        omegaiSample = omegai       
        
        for p in omegaiSample:
            #p = omegai[indsP[j]] 
            q = getNonZeroRow(X, i, n) 
        
            uivp = dot(U, i, V, p, k)
            uivq = dot(U, i, V, q, k)
            
            gamma = uivp - uivq
            zeta = 1 - gamma - xi[i]
            
            if zeta > 0: 
                deltaTheta -= zeta 
            
        deltaTheta /= float(omegaiSample.shape[0] * m)
    
    deltaTheta += C/m 

    return deltaTheta

@cython.boundscheck(False)
@cython.wraparound(False)
def updateUVApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=2, mode="c"] muU, numpy.ndarray[double, ndim=2, mode="c"] muV, numpy.ndarray[double, ndim=1, mode="c"] xi, numpy.ndarray[double, ndim=1, mode="c"] muXi, list omegaList, list omegaProbabilitiesList, numpy.ndarray[unsigned int, ndim=1, mode="c"] rowInds,  numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, unsigned int ind, double sigma, unsigned int numIterations, unsigned int numRowSamples, unsigned int numAucSamples, double w, double lmbda, double C, bint normalise): 
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]    
    cdef unsigned int k = U.shape[1] 
    cdef double normUi
    cdef numpy.ndarray[double, ndim=1, mode="c"] dUi = numpy.zeros(k)
    cdef numpy.ndarray[double, ndim=1, mode="c"] dVj = numpy.zeros(k)
    cdef unsigned int i, j, s, ind2, dXii
    cdef unsigned int startAverage = 10
    
    for s in range(numIterations):
        i = rowInds[(ind + s) % m]
        dUi = derivativeUiApprox(X, U, V, omegaList, omegaProbabilitiesList, i, numRowSamples, numAucSamples, xi, lmbda, C, normalise)
        #dUi = derivativeUi(X, U, V, omegaList, i, r, nu)
        
        j = colInds[(ind + s) % n]
        dVj = derivativeViApprox(X, U, V, omegaList, omegaProbabilitiesList, j, numRowSamples, numAucSamples, xi, lmbda, C, normalise)
        #dVi = derivativeVi(X, U, V, omegaList, j, r, nu)

        dXii = derivativeXiiApprox(X, U, V, omegaList, omegaProbabilitiesList, i, numRowSamples, numAucSamples, xi, lmbda, C, normalise)

        plusEquals(U, i, -sigma*dUi, k)
        
        normUi = numpy.linalg.norm(U[i,:])
        #normUi = normRow(U, i, k)
        
        if normUi != 0: 
            U[i,:] = scale(U, i, 1/normUi, k)             
        
        plusEquals(V, j, -sigma*dVj, k)
        
        #normVj = numpy.linalg.norm(V[j,:])
        #if normVj > 1: 
        #    V[j,:] = scale(V, j, 1/normVj, k)  
        
        xi[i] -= sigma*dXii

        if xi[i] < 0: 
            xi[i] = 0
        
        ind2 = ind/m
        
        if ind2 > startAverage: 
            muU[i, :] = muU[i, :]*ind2/float(ind2+1) + U[i, :]/float(ind2+1)
            muV[j, :] = muV[j, :]*ind2/float(ind2+1) + V[j, :]/float(ind2+1)
            muXi[i] = muXi[i]*ind2/float(ind2+1) + xi[i]/float(ind2+1)
        else: 
            muU[i, :] = U[i, :]
            muV[j, :] = V[j, :]
            muXi[i] = xi[i]
            
        
@cython.boundscheck(False)
@cython.wraparound(False)   
def objectiveApprox(X, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, list omegaList, unsigned int numAucSamples, numpy.ndarray[double, ndim=1, mode="c"] xi, double lmbda, double C):         
    cdef double obj = 0 
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]
    cdef unsigned int i, j, k, p, q
    cdef double kappa, uivp, uivq, gamma, partialObj
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegai = numpy.zeros(10, numpy.uint)  
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaiSample = numpy.zeros(10, numpy.uint)
    #cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsP
    
    k = U.shape[1]
    
    for i in range(m): 
        omegai = omegaList[i]
        #omegaBari = numpy.setdiff1d(numpy.arange(n), omegai, assume_unique=True)
        
        if omegai.shape[0] * (n-omegai.shape[0]) != 0: 
            partialObj = 0                
            
            #indsP = numpy.random.randint(0, omegai.shape[0], numAucSamples)  
            #indsQ = numpy.random.randint(0, omegaBari.shape[0], numAucSamples)
            omegaiSample = numpy.random.choice(omegai, numAucSamples) 
            
            for p in omegaiSample:
                #p = omegai[indsP[j]] 
                #q = omegaBari[indsQ[j]]
                q = getNonZeroRow(X, i, n)                  
            
                uivp = dot(U, i, V, p, k)
                gamma = uivp - uivq
                
                uivq = dot(U, i, V, q, k)
                
                if gamma + xi[i] <= 1: 
                    partialObj += ((1-gamma-xi[i])**2) 
                
            obj += partialObj/float(omegaiSample.shape[0])
    
    obj /= 2*m       
    obj += (lmbda/(2*m))*numpy.linalg.norm(V)**2 + C*numpy.sum(xi)
    
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
    cdef numpy.ndarray[numpy.uint_t, ndim=1, mode="c"] omegaiSample = numpy.zeros(10, numpy.uint)
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

            omegaiSample = numpy.random.choice(omegai, numAucSamples)            
            
            for p in omegaiSample:
                #ind = numpy.random.randint(omegai.shape[0])
                #ind = randint(nOmegai)
                #p = omegai[ind] 
                
                q = getNonZeroRow(X, i, n)                
                uivp = dot(U, i, V, p, k)

                if uivp > ri and uivp > dot(U, i, V, q, k): 
                    partialAuc += 1 
                        
            localAucArr[i] = partialAuc/float(omegaiSample.shape[0])     
    
    return localAucArr.mean()