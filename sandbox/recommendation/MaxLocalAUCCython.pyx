#cython: profile=True 
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
from __future__ import print_function
import cython
from cython.parallel import parallel, prange
cimport numpy
import numpy
from sandbox.util.CythonUtils cimport dot, scale, choice, inverseChoice, uniformChoice, plusEquals, partialSum, square
from sandbox.util.SparseUtilsCython import SparseUtilsCython


from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX

cdef extern from "math.h":
    double exp(double x)
    double tanh(double x)
    bint isnan(double x)  
    double sqrt(double x)

cdef computeOmegaProbs(unsigned int i, numpy.ndarray[int, ndim=1, mode="c"] omegai, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V): 
    cdef numpy.ndarray[double, ndim=1, mode="c"] uiVOmegai
    cdef numpy.ndarray[double, ndim=1, mode="c"] colIndsCumProbs
    cdef double ri, z = 5
    
    uiVOmegai = U[i, :].T.dot(V[omegai, :].T)
    #Check this line when omegai.shape < z 
    ri = numpy.sort(uiVOmegai)[-min(z, uiVOmegai.shape[0])]
    colIndsCumProbs = numpy.array(uiVOmegai >= ri, numpy.float)
    colIndsCumProbs /= colIndsCumProbs.sum()
    colIndsCumProbs = numpy.cumsum(colIndsCumProbs)
    
    return colIndsCumProbs

cdef inline itemRank(numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[int, ndim=1, mode="c"] omegai, unsigned int i, double uivp, unsigned int numOmegaBari): 
    """
    Use the sampling scheme from k-order paper to get an estimate of the rank of an item 
    """

    cdef unsigned int rank = 0
    cdef unsigned int k = U.shape[1]
    cdef unsigned int n = V.shape[0]
    cdef unsigned int q
    cdef double uivq 

    while True: 
        q = inverseChoice(omegai, n) 
        rank += 1
        uivq = dot(U, i, V, q, k)
        
        if rank >= numOmegaBari or uivq <= uivp - 1: 
            break 
        
    return rank+1

def derivativeUi(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r,  numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, unsigned int i, double rho, bint normalise):
    """
    Find  delta phi/delta u_i using the hinge loss.  
    """
    cdef unsigned int p, q
    cdef unsigned int k = U.shape[1]
    cdef double uivp, uivq, gamma, kappa, ri
    cdef double  normDeltaTheta, hGamma, hKappa, vpScale, normGp, normGq 
    cdef unsigned int m = U.shape[0], n = V.shape[0], numOmegai, numOmegaBari
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaBari 
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
      
    omegai = colInds[indPtr[i]:indPtr[i+1]]
    omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.int32), omegai, assume_unique=True)
    numOmegai = omegai.shape[0]
    numOmegaBari = n-numOmegai
    
    deltaTheta = numpy.zeros(k)
    ri = r[i]
    normGp = gp[omegai].sum()
    normGq = gq[omegaBari].sum()
    
    if numOmegai * numOmegaBari != 0:         
        for p in omegai: 
            vpScale = 0
            uivp = dot(U, i, V, p, k)
            kappa = rho*(uivp - ri)
            hKappa = 1-kappa
            
            for q in omegaBari: 
                uivq = dot(U, i, V, q, k)
                
                gamma = uivp - uivq
                hGamma = 1-gamma 
                
                if hGamma > 0 and hKappa > 0: 
                    #vpScale -= hGamma*(hKappa**2) + (hGamma**2)*hKappa*rho
                    #deltaTheta += V[q, :]*hGamma*(hKappa**2)
                    deltaTheta += gp[p] * gq[q] * ((V[q, :] - V[p, :])*hGamma*tanh(hKappa) - V[p, :]*(rho/2)*(hGamma**2)*(1 - tanh(hKappa)**2))/(normGp*normGq)
                
        deltaTheta *= gi[i]
                    
    #Normalise gradient to have unit norm 
    normDeltaTheta = numpy.linalg.norm(deltaTheta)
    
    if normDeltaTheta != 0 and normalise: 
        deltaTheta = deltaTheta/normDeltaTheta
    
    return deltaTheta

def updateU(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r, double sigma, double rho, bint normalise):  
    """
    Compute the full gradient descent update of U
    """    
    
    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dU = numpy.zeros((U.shape[0], U.shape[1]), numpy.float)
    cdef unsigned int i 
    cdef unsigned int m = U.shape[0]
    cdef unsigned int k = U.shape[1]
    
    for i in range(m): 
        dU[i, :] = derivativeUi(indPtr, colInds, U, V, r, i, rho, normalise) 
    
    U -= sigma*dU
    
    for i in range(m):
        U[i,:] = scale(U, i, 1/numpy.linalg.norm(U[i,:]), k)   


def derivativeUiApprox(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V,  numpy.ndarray[double, ndim=1, mode="c"] r,  numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, unsigned int i, unsigned int numRowSamples, unsigned int numAucSamples, double rho, double lmbda, bint normalise):
    """
    Find an approximation of delta phi/delta u_i using the simple objective without 
    sigmoid functions. 
    """
    cdef unsigned int p, q, ind, j, s
    cdef unsigned int k = U.shape[1]
    cdef double uivp, ri, uivq, gamma, kappa, hGamma, hKappa
    cdef double normDeltaTheta, vqScale, vpScale, normGp=0, normGpq=0, tanhHKappa, rhoOver2 = rho/2
    cdef unsigned int m = U.shape[0], n = V.shape[0], numOmegai, numOmegaBari
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaiSample
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ = numpy.zeros(k, numpy.int)
         
    omegai = colInds[indPtr[i]:indPtr[i+1]]
    numOmegai = omegai.shape[0]
    numOmegaBari = n-numOmegai
    ri = r[i]
    
    deltaTheta = numpy.zeros(k)
    
    if numOmegai * numOmegaBari != 0: 
        omegaiSample = uniformChoice(omegai, numAucSamples)   
        
        for p in omegaiSample:
            q = inverseChoice(omegai, n) 
        
            uivp = dot(U, i, V, p, k)
            uivq = dot(U, i, V, q, k)
            
            gamma = uivp - uivq
            kappa = rho*(uivp - ri) 
            hGamma = max(1 - gamma, 0)
            hKappa = max(1 - kappa, 0)
            
            zeta = gp[p]*gq[q]
            normGpq += zeta
            tanhHKappa = tanh(hKappa)           
            
            if hGamma > 0 and hKappa > 0: 
                vqScale = zeta*hGamma*tanhHKappa
                vpScale = -vqScale - zeta*rhoOver2*square(hGamma)*(1 - square(tanhHKappa))
                
                deltaTheta += scale(V, p, vpScale, k) + scale(V, q, vqScale, k)    
                
        #if normGp*normGq != 0: 
        deltaTheta *= gi[i]/normGpq
        
    deltaTheta += scale(U, i, lmbda/m, k)
                    
    #Normalise gradient to have unit norm 
    normDeltaTheta = numpy.linalg.norm(deltaTheta)
    
    if normDeltaTheta != 0 and normalise: 
        deltaTheta = deltaTheta/normDeltaTheta
    
    return deltaTheta


def derivativeVi(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r, numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, unsigned int j, double rho, bint normalise): 
    """
    delta phi/delta v_i using hinge loss. 
    """
    cdef unsigned int i = 0
    cdef unsigned int k = U.shape[1]
    cdef unsigned int p, q, numOmegai, numOmegaBari, t
    cdef unsigned int m = U.shape[0]
    cdef unsigned int n = V.shape[0], ind
    cdef unsigned int s = 0
    cdef double uivp, uivq,  betaScale, ri, normTheta, gamma, kappa, hGamma, hKappa, normGp, normGq
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaBari
    
    for i in range(m): 
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.int32), omegai, assume_unique=True)
        numOmegai = omegai.shape[0]       
        numOmegaBari = n-numOmegai
        ri = r[i]
        
        betaScale = 0
        normGp = 0
        normGq = 0
        
        if j in omegai:                 
            p = j 
            uivp = dot(U, i, V, p, k)
            
            normGp = gp[omegai].sum()
            
            for q in omegaBari: 
                uivq = dot(U, i, V, q, k)
                gamma = uivp - uivq
                kappa = rho*(uivp - ri)
                
                hGamma = 1-gamma 
                hKappa = 1-kappa
                
                normGq += gq[q]

                if hGamma > 0 and hKappa>0: 
                    #betaScale += hGamma*hKappa**2 + hGamma**2*hKappa*rho
                    betaScale += gp[p] * gq[q] * (hGamma*tanh(hKappa) + (rho/2)*hGamma**2 * (1- tanh(hKappa)**2))
            
            if normGp*normGq != 0: 
                deltaBeta = scale(U, i, -betaScale/(normGp*normGq), k)
        else:
            q = j 
            uivq = dot(U, i, V, q, k)
            
            normGq = gq[omegaBari].sum()
                            
            for p in omegai: 
                uivp = dot(U, i, V, p, k)
                gamma = uivp - uivq  
                kappa = rho*(uivp - ri)
                
                hGamma = 1-gamma 
                hKappa = 1-kappa
                
                normGp += gp[p]
                
                if hGamma > 0 and hKappa>0:   
                    #betaScale += hGamma*hKappa**2
                    betaScale += gp[p] * gq[q] * hGamma * tanh(hKappa)

            if normGp*normGq != 0: 
                deltaBeta = scale(U, i, betaScale/(normGp*normGq), k)  
                
        deltaTheta += deltaBeta * gi[i]
    
    deltaTheta /= gi.sum()
       
    
    #Make gradient unit norm 
    normTheta = numpy.linalg.norm(deltaTheta)
    if normTheta != 0 and normalise: 
        deltaTheta = deltaTheta/normTheta
    
    return deltaTheta
 

def updateV(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r, double sigma, double lmbda, double rho, bint normalise): 
    """
    Compute the full gradient descent update of V
    """
    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dV = numpy.zeros((V.shape[0], V.shape[1]), numpy.float)
    cdef unsigned int j
    cdef unsigned int n = V.shape[0]
    cdef unsigned int k = V.shape[1]
    
    for j in range(n): 
        dV[j, :] = derivativeVi(indPtr, colInds, U, V, r, j, lmbda, rho, normalise) 
        
    V -= sigma*dV
    
    for j in range(n): 
        normVj = numpy.linalg.norm(V[j,:])        
        if normVj >= lmbda: 
            V[j,:] = scale(V, j, lmbda/normVj, k)               

def derivativeViApprox(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r,  numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[double, ndim=1, mode="c"] normGp, numpy.ndarray[double, ndim=1, mode="c"] normGq, unsigned int j, unsigned int numRowSamples, unsigned int numAucSamples, double rho, double lmbda, bint normalise): 
    """
    delta phi/delta v_i  using the hinge loss. 
    """
    cdef unsigned int i = 0
    cdef unsigned int k = U.shape[1]
    cdef unsigned int p, q, numOmegai
    cdef unsigned int m = U.shape[0]
    cdef unsigned int n = V.shape[0]
    cdef unsigned int s = 0
    cdef double uivp, uivq,  betaScale, normTheta, gamma, kappa, nu, nuPrime, hGamma, hKappa, zeta, ri, normBeta, normGqi, normGpi, rhoOver2
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] rowInds = numpy.random.permutation(m)[0:numRowSamples]
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaiSample
    
    rhoOver2 = rho/2    
    
    for i in rowInds: 
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        numOmegai = omegai.shape[0]       
        
        betaScale = 0
        ri = r[i]
        normBeta = 0
        normGqi = 0
        normGpi = 0 
        
        if j in omegai:                 
            p = j 
            uivp = dot(U, i, V, p, k)
            nu = 1 - uivp
            hKappa = max(0, 1 - rho*(uivp - ri))
            zeta = tanh(hKappa)
            #normGp = partialSum(gp, omegai)

            for s in range(numAucSamples): 
                q = inverseChoice(omegai, n)
                uivq = dot(U, i, V, q, k)                
                hGamma = max(0, nu+uivq) 
                normGqi += gq[q]
                
                betaScale += gp[p] * gq[q] * (hGamma*zeta + rhoOver2*square(hGamma) * (1 - square(zeta)))  
                             
            if normGp[i]*normGqi != 0:
                deltaBeta = scale(U, i, -betaScale/(normGp[i]*normGqi), k)
        elif numOmegai != 0:
            q = j 
            uivq = dot(U, i, V, q, k)
            nu = 1 + uivq 
            nuPrime = 1 + ri*rho
            omegaiSample = uniformChoice(omegai, numAucSamples)

            for p in omegaiSample: 
                uivp = dot(U, i, V, p, k)
                hGamma = max(0, nu - uivp) 
                hKappa = max(0, nuPrime - rho*uivp)
                normGpi += gp[p]
                
                betaScale += gp[p] * gq[q]*hGamma*tanh(hKappa)
            
            if normGpi*normGq[i] != 0:
                deltaBeta = scale(U, i, betaScale/(normGpi*normGq[i]), k)  
                
        deltaTheta += deltaBeta*gi[i]
        
    deltaTheta /= gi[rowInds].sum()
    deltaTheta += scale(V, j, lmbda/n, k)
    
    #Make gradient unit norm 
    normTheta = numpy.linalg.norm(deltaTheta)
    if normTheta != 0 and normalise: 
        deltaTheta = deltaTheta/normTheta
    
    return deltaTheta

def meanPositive(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] V, unsigned int i): 
    """
    Compute u_i = 1/omegai \sum_p \in \omegai vp i.e. the mean positive item. 
    """
    cdef unsigned int p 
    cdef unsigned int k = V.shape[1]
    cdef numpy.ndarray[double, ndim=1, mode="c"] ui = numpy.zeros(k)
    
    omegai = colInds[indPtr[i]:indPtr[i+1]]
    
    for p in omegai: 
        ui += V[p, :]
    
    if omegai.shape[0] != 0:
        ui /= omegai.shape[0]
    
    return ui
    

def updateUVApprox(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=2, mode="c"] muU, numpy.ndarray[double, ndim=2, mode="c"] muV, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedRowInds,  numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[double, ndim=1, mode="c"] normGp, numpy.ndarray[double, ndim=1, mode="c"] normGq, unsigned int ind, double sigma, unsigned int numRowSamples, unsigned int numAucSamples, double w, double lmbdaU, double lmbdaV, double rho, unsigned int startAverage, unsigned int numIterations, bint normalise, bint itemFactors): 
    cdef unsigned int m = U.shape[0]
    cdef unsigned int n = V.shape[0]    
    cdef unsigned int k = U.shape[1] 
    cdef unsigned int i, j, s
    cdef unsigned int printStep = 1000
    cdef double normUi, normVj, maxNorm=100
    cdef numpy.ndarray[double, ndim=1, mode="c"] dUi = numpy.zeros(k)
    cdef numpy.ndarray[double, ndim=1, mode="c"] dVj = numpy.zeros(k)
    cdef numpy.ndarray[double, ndim=1, mode="c"] r 

    for s in range(numIterations):
        if s % printStep == 0: 
            print(str(s) + " ", end="")
            
        r = SparseUtilsCython.computeR(U, V, w, numAucSamples)
        
        i = permutedRowInds[s % permutedRowInds.shape[0]]   
        
        if itemFactors: 
            U[i,:] = meanPositive(indPtr, colInds, V, i)
        else: 
            dUi = derivativeUiApprox(indPtr, colInds, U, V, r, gi, gp, gq, i, numRowSamples, numAucSamples, rho, lmbdaU, normalise)
            plusEquals(U, i, -sigma*dUi, k)
            normUi = numpy.linalg.norm(U[i,:])
            
            if normUi >= maxNorm: 
                U[i,:] = scale(U, i, maxNorm/normUi, k)             
        
        if ind > startAverage: 
            muU[i, :] = muU[i, :]*ind/float(ind+1) + U[i, :]/float(ind+1)
        else: 
            muU[i, :] = U[i, :]
            
        #Now update V
        #r = SparseUtilsCython.computeR(U, V, w, numAucSamples)        
        j = permutedColInds[s % permutedColInds.shape[0]]
        dVj = derivativeViApprox(indPtr, colInds, U, V, r, gi, gp, gq, normGp, normGq, j, numRowSamples, numAucSamples, rho, lmbdaV, normalise)
        plusEquals(V, j, -sigma*dVj, k)
        normVj = numpy.linalg.norm(V[j,:])  
        
        if normVj >= maxNorm: 
            V[j,:] = scale(V, j, maxNorm/normVj, k)        
        
        if ind > startAverage: 
            muV[j, :] = muV[j, :]*ind/float(ind+1) + V[j, :]/float(ind+1)
        else: 
            muV[j, :] = V[j, :]
           
def objectiveApprox(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[int, ndim=1, mode="c"] allIndPtr, numpy.ndarray[int, ndim=1, mode="c"] allColInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r,   numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, unsigned int numAucSamples, double rho, bint full=False):         
    cdef unsigned int m = U.shape[0]
    cdef unsigned int n = V.shape[0]
    cdef unsigned int i, j, k, p, q
    cdef double uivp, uivq, gamma, kappa, ri, partialObj, hGamma, hKappa, normGp, normGq, normGi=0, zeta, normGpq
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] allOmegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaiSample
    cdef numpy.ndarray[double, ndim=1, mode="c"] objVector = numpy.zeros(m, dtype=numpy.float)

    k = U.shape[1]
    
    for i in range(m): 
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        allOmegai = allColInds[allIndPtr[i]:allIndPtr[i+1]]
        
        ri = r[i]
        normGpq = 0 
        normGi += gi[i]
        partialObj = 0                
        
        omegaiSample = uniformChoice(omegai, numAucSamples) 
        #omegaiSample = omegai
        
        for p in omegaiSample:
            q = inverseChoice(allOmegai, n)                  
        
            uivp = dot(U, i, V, p, k)
            uivq = dot(U, i, V, q, k)
            
            gamma = uivp - uivq
            hGamma = max(0, 1 - gamma)
                            
            kappa = rho*(uivp - ri)
            hKappa = max(0, 1 - kappa)
            
            zeta = gp[p]*gq[q]
            normGpq += zeta
            
            partialObj += zeta * square(hGamma) * tanh(hKappa)
        
        if normGpq != 0: 
            objVector[i] = partialObj*gi[i]/normGpq
    
    objVector /= 2*normGi       
    
    if full: 
        return objVector
    else: 
        return objVector.sum() 
  
def objective(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[int, ndim=1, mode="c"] allIndPtr, numpy.ndarray[int, ndim=1, mode="c"] allColInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r,  numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, double rho, bint full=False):         
    """
    Note that distributions gp, gq and gi must be normalised to have sum 1. 
    """
        
    cdef unsigned int m = U.shape[0]
    cdef unsigned int n = V.shape[0]
    cdef unsigned int i, j, k, p, q
    cdef double uivp, uivq, gamma, kappa, ri, hGamma, hKappa, normGp, normGq, normGi=0, sumQ=0
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaBari 
    cdef numpy.ndarray[int, ndim=1, mode="c"] allOmegai 
    cdef numpy.ndarray[double, ndim=1, mode="c"] objVector = numpy.zeros(m, dtype=numpy.float)

    k = U.shape[1]
    
    for i in range(m): 
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        allOmegai = allColInds[allIndPtr[i]:allIndPtr[i+1]]
        ri = r[i]
        
        omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.int32), omegai, assume_unique=True)

        partialObj = 0 
        normGp = 0
        
        normGi += gi[i]
                    
        for p in omegai:
            uivp = dot(U, i, V, p, k)
            
            kappa = rho*(uivp - ri)
            hKappa = max(0, 1-kappa)
            normGp += gp[p]
            
            normGq = 0 
            sumQ = 0 
            
            for q in omegaBari:                 
                uivq = dot(U, i, V, q, k)
                gamma = uivp - uivq
                hGamma = max(0, 1-gamma)
                
                normGq += gq[q]

                sumQ += gp[p]*gq[q]*hGamma**2 * tanh(hKappa)
                    
            partialObj += sumQ/normGq
            
        objVector[i] = gi[i]*partialObj/normGp
    
    objVector /= 2*normGi        
    
    if full: 
        return objVector
    else: 
        return objVector.sum() 
