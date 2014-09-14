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


from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX

cdef extern from "math.h":
    double exp(double x)
    double tanh(double x)
    bint isnan(double x)  
    double sqrt(double x)
    double fmax(double x, double y)

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

cdef itemRank(numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[int, ndim=1, mode="c"] omegai, unsigned int i, double uivp, unsigned int numOmegaBari): 
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


cdef class MaxLocalAUCCython(object): 
    cdef public unsigned int k, printStep, numAucSamples, numRowSamples, startAverage
    cdef public double lmbdaU, lmbdaV, maxNorm, rho, w
    cdef public bint itemFactors, normalise
    
    def __init__(self, unsigned int k=8, double lmbdaU=0.0, double lmbdaV=1.0, bint normalise=True, unsigned int numAucSamples=10, unsigned int numRowSamples=30, unsigned int startAverage=30, double rho=0.5, double w=0.9): 
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
        self.w = w 
    
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
                
                zeta += gq[q]*square(hGamma)
                normGq += gq[q]
                
                deltaBeta += (V[q, :] - V[p, :])*gq[q]*hGamma
             
            deltaTheta += deltaBeta*(1 - tanh(zeta/normGq)**2)*gp[p]/normGq
        
        if normGp != 0:
            deltaTheta /= m*normGp
        deltaTheta += scale(U, i, self.lmbdaU/m, self.k)
                    
        #Normalise gradient to have unit norm 
        normDeltaTheta = numpy.linalg.norm(deltaTheta)
        
        if normDeltaTheta != 0 and self.normalise: 
            deltaTheta = deltaTheta/normDeltaTheta
        
        return deltaTheta
    
    def updateU(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, double sigma):  
        """
        Compute the full gradient descent update of U
        """    
        
        cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dU = numpy.zeros((U.shape[0], U.shape[1]), numpy.float)
        cdef unsigned int i 
        cdef unsigned int m = U.shape[0]
        
        for i in range(m): 
            dU[i, :] = self.derivativeUi(indPtr, colInds, U, V, gp, gq, i) 
        
        U -= sigma*dU
        
    def derivativeUiApprox(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, unsigned int i):
        """
        Find an approximation of delta phi/delta u_i using the simple objective without 
        sigmoid functions. 
        """
        cdef unsigned int p, q, ind, j, s
        cdef double uivp, uivq, gamma, kappa, hGamma,
        cdef double normDeltaTheta, normGp, normGq, zeta, nu
        cdef unsigned int m = U.shape[0], n = V.shape[0]
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaiSample
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(self.k, numpy.float)
        cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ = numpy.zeros(self.k, numpy.int)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(self.k, numpy.float)
             
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        
        #This ought to restrict omega to permutedColInds
        #omegaiSample = numpy.intersect1d(omegai, permutedColInds, assume_unique=True)                
        omegaiSample = uniformChoice(omegai, self.numAucSamples)   
        normGp = 0
        
        for p in omegaiSample: 
            uivp = dot(U, i, V, p, self.k)
            normGp += gp[p]
            
            deltaBeta = numpy.zeros(self.k, numpy.float)
            kappa = 0
            zeta = 0 
            normGq = 0
            
            for j in range(self.numAucSamples): 
                q = inverseChoiceArray(omegai, permutedColInds) 
                uivq = dot(U, i, V, q, self.k)
                
                gamma = uivp - uivq
                hGamma = max(0, 1-gamma) 
                
                nu = gq[q]*hGamma
                zeta += nu*hGamma
                normGq += gq[q]
                
                
                #deltaBeta += (V[q, :] - V[p, :])*(gq[q]*hGamma)
                deltaBeta += scale(V, q, nu, self.k) - scale(V, p, nu, self.k)
             
            deltaTheta += deltaBeta*(1 - tanh(zeta/normGq)**2)*gp[p]/normGq
         
        if normGp != 0:
            deltaTheta /= m*normGp
        deltaTheta += scale(U, i, self.lmbdaU/m, self.k)
                        
        #Normalise gradient to have unit norm 
        if self.normalise: 
            normDeltaTheta = numpy.linalg.norm(deltaTheta)
            
            if normDeltaTheta != 0: 
                deltaTheta = deltaTheta/normDeltaTheta
        
        return deltaTheta
    
    def derivativeUiApprox2(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, unsigned int i):
        """
        Find an approximation of delta phi/delta u_i using the simple objective without 
        sigmoid functions. 
        """
        cdef unsigned int p, q, ind, j, s
        cdef double uivp, uivq, gamma, kappa, hGamma,
        cdef double normDeltaTheta, normGp, normGq, zeta, nu
        cdef unsigned int m = U.shape[0], n = V.shape[0]
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaiSample
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(self.k, numpy.float)
        cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ = numpy.zeros(self.k, numpy.int)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(self.k, numpy.float)
             
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        
        #This ought to restrict omega to permutedColInds
        #omegaiSample = numpy.intersect1d(omegai, permutedColInds, assume_unique=True)                
        omegaiSample = uniformChoice(omegai, self.numAucSamples)   
        normGp = 0
        
        for p in omegaiSample: 
            uivp = dot(U, i, V, p, self.k)
            normGp += gp[p]
            
            deltaBeta = numpy.zeros(self.k, numpy.float)
            kappa = 0
            zeta = 0 
            normGq = 0
            
            for j in range(self.numAucSamples): 
                q = inverseChoiceArray(omegai, permutedColInds) 
                uivq = dot(U, i, V, q, self.k)
                
                gamma = uivp - uivq
                hGamma = max(0, 1-gamma) 
                
                nu = gq[q]*hGamma
                zeta += nu*hGamma
                normGq += gq[q]
                
                
                #deltaBeta += (V[q, :] - V[p, :])*(gq[q]*hGamma)
                deltaBeta += scale(V, q, nu, self.k) - scale(V, p, nu, self.k)
             
            deltaTheta += deltaBeta*gp[p]/normGq
         
        if normGp != 0:
            deltaTheta /= m*normGp
        deltaTheta += scale(U, i, self.lmbdaU/m, self.k)
                        
        #Normalise gradient to have unit norm 
        if self.normalise: 
            normDeltaTheta = numpy.linalg.norm(deltaTheta)
            
            if normDeltaTheta != 0: 
                deltaTheta = deltaTheta/normDeltaTheta
        
        return deltaTheta    
    
    
    def derivativeVi(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, unsigned int j): 
        """
        delta phi/delta v_i using hinge loss. 
        """
        cdef unsigned int i = 0
        cdef unsigned int k = U.shape[1]
        cdef unsigned int p, q, numOmegai, numOmegaBari, t, ell
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0], ind
        cdef unsigned int s = 0
        cdef double uivp, uivq,  betaScale, ri, normTheta, gamma, kappa, hGamma, hKappa, normGp, normGq
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaBari
        
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint32), omegai, assume_unique=True)
            
            betaScale = 0
            normGp = 0
            normGq = 0
            
            if j in omegai:                 
                p = j 
                uivp = dot(U, i, V, p, k)
                
                normGp = gp[omegai].sum()
                normGq = 0
                zeta = 0
                kappa = 0 
                
                for q in omegaBari: 
                    uivq = dot(U, i, V, q, k)
                    gamma = uivp - uivq
                    hGamma = max(0, 1-gamma) 
                    
                    kappa += gq[q]*hGamma
                    zeta += gq[q]*square(hGamma)
                    normGq += gq[q]
                
                if normGq != 0: 
                    kappa /= normGq
                    zeta /= normGq
                    
                if normGp != 0: 
                    betaScale -= kappa*(1 - tanh(zeta)**2)*gp[p]/normGp
            else:
                q = j 
                uivq = dot(U, i, V, q, k)
                
                normGp = 0 
                normGq = gq[omegaBari].sum()
                kappa = 0
                
                for p in omegai: 
                    uivp = dot(U, i, V, p, k)
                    gamma = uivp - uivq  
                    hGamma = max(0, 1-gamma) 
                    zeta = 0
                    
                    for ell in omegaBari:
                        uivell = dot(U, i, V, ell, k)
                        gamma2 = uivp - uivell  
                        hGamma2 = max(0, 1-gamma2)
                        zeta += gq[ell]*square(hGamma2)
                    
                    if normGq != 0: 
                        zeta /= normGq
                    
                    kappa += gp[p]*gq[q]*hGamma * (1- tanh(zeta)**2)
                    normGp += gp[p]                    
                    
                if normGp*normGq != 0: 
                    betaScale += kappa/(normGp*normGq)
            
            #print(betaScale, U[i, :])
            deltaTheta += U[i, :]*betaScale 
        
        deltaTheta /= m
        deltaTheta += scale(V, j, self.lmbdaV/m, self.k)
        
        #Make gradient unit norm 
        normTheta = numpy.linalg.norm(deltaTheta)
        if normTheta != 0 and self.normalise: 
            deltaTheta = deltaTheta/normTheta
        
        return deltaTheta
     
    
    def updateV(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, double sigma): 
        """
        Compute the full gradient descent update of V
        """
        cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dV = numpy.zeros((V.shape[0], V.shape[1]), numpy.float)
        cdef unsigned int j
        cdef unsigned int n = V.shape[0]
        cdef unsigned int k = V.shape[1]
        
        for j in range(n): 
            dV[j, :] = self.derivativeVi(indPtr, colInds, U, V, gp, gq, j) 
            
        V -= sigma*dV
                  
    
    def derivativeViApprox(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[double, ndim=1, mode="c"] normGp, numpy.ndarray[double, ndim=1, mode="c"] normGq, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedRowInds,  numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, unsigned int j): 
        """
        delta phi/delta v_i  using the hinge loss. 
        """
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0]
        cdef unsigned int i, p, q, s, ell
        cdef double uivp, uivq, uivell, betaScale, normTheta, gamma, kappa, hGamma, zeta, normGqi, normGpi, gamma2, hGamma2, nu
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(self.k, numpy.float)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(self.k, numpy.float)
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] rowInds = numpy.random.choice(permutedRowInds, min(self.numRowSamples, permutedRowInds.shape[0]), replace=False)
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaiSample
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaBari
        cdef numpy.ndarray[double, ndim=1, mode="c"] uivqs = numpy.zeros(self.numAucSamples, numpy.float)
        
        for i in rowInds:
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            #omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint32), omegai, assume_unique=True)
            
            #Find an array not in omega but in permutedColInds 
            omegaBari = numpy.zeros(self.numAucSamples, numpy.uint32)
            for s in range(self.numAucSamples): 
                omegaBari[s] = inverseChoiceArray(omegai, permutedColInds)
                #Can compute uivqs here 
                uivqs[s] = dot(U, i, V, omegaBari[s], self.k)

            betaScale = 0
            
            if j in omegai:                 
                p = j 
                uivp = dot(U, i, V, p, self.k)
                
                normGqi = 0
                zeta = 0
                kappa = 0 
                
                for s, q in enumerate(omegaBari): 
                    #uivq = dot(U, i, V, q, k)
                    uivq = uivqs[s]
                    gamma = uivp - uivq
                    hGamma = max(0, 1-gamma) 
                    
                    nu = gq[q]*hGamma                    
                    
                    kappa += nu
                    zeta += nu*hGamma
                    normGqi += gq[q]
                
                if normGqi != 0: 
                    kappa /= normGqi
                    zeta /= normGqi
                    
                if normGp[i] != 0: 
                    betaScale -= kappa*(1 - square(tanh(zeta)))*gp[p]/normGp[i]
            else:
                q = j 
                uivq = dot(U, i, V, q, self.k)
                
                normGpi = 0 
                kappa = 0
                
                #This ought to restrict omega to permutedColInds
                #omegaiSample = numpy.intersect1d(omegai, permutedColInds, assume_unique=True)
                omegaiSample = uniformChoice(omegai, self.numAucSamples)
                
                for p in omegaiSample: 
                    #for p in omegai: 
                    uivp = dot(U, i, V, p, self.k)
                    gamma = uivp - uivq  
                    hGamma = max(0, 1-gamma) 
                    zeta = 0
                    
                    for s, ell in enumerate(omegaBari): 
                        #uivell = dot(U, i, V, ell, k)
                        uivell = uivqs[s]
                        gamma2 = uivp - uivell  
                        hGamma2 = max(0, 1-gamma2)
                        zeta += gq[ell]*square(hGamma2)
                    
                    if normGq[i] != 0: 
                        zeta /= normGq[i]
                    
                    kappa += gp[p]*gq[q]*hGamma * (1- square(tanh(zeta)))
                    normGpi += gp[p]                    
                    
                if normGp[i]*normGpi != 0: 
                    betaScale += kappa/(normGpi*normGq[i])
            
            deltaTheta += scale(U, i, betaScale, self.k) 
        
        deltaTheta /= rowInds.shape[0]
        deltaTheta += scale(V, j, self.lmbdaV/m, self.k)
        
        #Make gradient unit norm
        if self.normalise: 
            normTheta = numpy.linalg.norm(deltaTheta)
            
            if normTheta != 0: 
                deltaTheta = deltaTheta/normTheta
        
        return deltaTheta

    def derivativeViApprox2(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[double, ndim=1, mode="c"] normGp, numpy.ndarray[double, ndim=1, mode="c"] normGq, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedRowInds,  numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, unsigned int j): 
        """
        delta phi/delta v_i  using the hinge loss. 
        """
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0]
        cdef unsigned int i, p, q, s, ell
        cdef double uivp, uivq, uivell, betaScale, normTheta, gamma, kappa, hGamma, zeta, normGqi, normGpi, gamma2, hGamma2, nu
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(self.k, numpy.float)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(self.k, numpy.float)
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] rowInds = numpy.random.choice(permutedRowInds, min(self.numRowSamples, permutedRowInds.shape[0]), replace=False)
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaiSample
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaBari
        cdef numpy.ndarray[double, ndim=1, mode="c"] uivqs = numpy.zeros(self.numAucSamples, numpy.float)
        
        for i in rowInds:
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            #omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint32), omegai, assume_unique=True)
            
            #Find an array not in omega but in permutedColInds 
            omegaBari = numpy.zeros(self.numAucSamples, numpy.uint32)
            for s in range(self.numAucSamples): 
                omegaBari[s] = inverseChoiceArray(omegai, permutedColInds)
                #Can compute uivqs here 
                uivqs[s] = dot(U, i, V, omegaBari[s], self.k)

            betaScale = 0
            
            if j in omegai:                 
                p = j 
                uivp = dot(U, i, V, p, self.k)
                
                normGqi = 0
                zeta = 0
                kappa = 0 
                
                for s, q in enumerate(omegaBari): 
                    #uivq = dot(U, i, V, q, k)
                    uivq = uivqs[s]
                    gamma = uivp - uivq
                    hGamma = max(0, 1-gamma) 
                    
                    nu = gq[q]*hGamma                    
                    
                    kappa += nu
                    zeta += nu*hGamma
                    normGqi += gq[q]
                
                if normGqi != 0: 
                    kappa /= normGqi
                    zeta /= normGqi
                    
                if normGp[i] != 0: 
                    betaScale -= kappa*gp[p]/normGp[i]
            else:
                q = j 
                uivq = dot(U, i, V, q, self.k)
                
                normGpi = 0 
                kappa = 0
                
                #This ought to restrict omega to permutedColInds
                #omegaiSample = numpy.intersect1d(omegai, permutedColInds, assume_unique=True)
                omegaiSample = uniformChoice(omegai, self.numAucSamples)
                
                for p in omegaiSample: 
                    #for p in omegai: 
                    uivp = dot(U, i, V, p, self.k)
                    gamma = uivp - uivq  
                    hGamma = max(0, 1-gamma) 
                    zeta = 0
                    
                    for s, ell in enumerate(omegaBari): 
                        #uivell = dot(U, i, V, ell, k)
                        uivell = uivqs[s]
                        gamma2 = uivp - uivell  
                        hGamma2 = max(0, 1-gamma2)
                        zeta += gq[ell]*square(hGamma2)
                    
                    if normGq[i] != 0: 
                        zeta /= normGq[i]
                    
                    kappa += gp[p]*gq[q]*hGamma 
                    normGpi += gp[p]                    
                    
                if normGp[i]*normGpi != 0: 
                    betaScale += kappa/(normGpi*normGq[i])
            
            deltaTheta += scale(U, i, betaScale, self.k) 
        
        deltaTheta /= rowInds.shape[0]
        deltaTheta += scale(V, j, self.lmbdaV/m, self.k)
        
        #Make gradient unit norm
        if self.normalise: 
            normTheta = numpy.linalg.norm(deltaTheta)
            
            if normTheta != 0: 
                deltaTheta = deltaTheta/normTheta
        
        return deltaTheta    
    
    def meanPositive(self, numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] V, unsigned int i): 
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
        
    
    def updateUVApprox(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=2, mode="c"] muU, numpy.ndarray[double, ndim=2, mode="c"] muV, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedRowInds,  numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[double, ndim=1, mode="c"] normGp, numpy.ndarray[double, ndim=1, mode="c"] normGq, unsigned int ind, unsigned int numIterations, double sigma): 
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0]    
        cdef unsigned int i, j, s
        cdef double normUi, normVj
        cdef bint newline = indPtr.shape[0] > 100000
        cdef numpy.ndarray[double, ndim=1, mode="c"] dUi = numpy.zeros(self.k)
        cdef numpy.ndarray[double, ndim=1, mode="c"] dVj = numpy.zeros(self.k)
    
        for s in range(numIterations):
            if s % self.printStep == 0: 
                if newline:  
                    print(str(s) + " of " + str(numIterations))
                else: 
                    print(str(s) + " ", end="")
                     
            i = permutedRowInds[s % permutedRowInds.shape[0]]   
            
            if self.itemFactors: 
                U[i,:] = self.meanPositive(indPtr, colInds, V, i)
            else: 
                dUi = self.derivativeUiApprox(indPtr, colInds, U, V, gp, gq, permutedColInds, i)
                plusEquals(U, i, -sigma*dUi, self.k)
                normUi = numpy.linalg.norm(U[i,:])
                
                if normUi >= self.maxNorm: 
                    U[i,:] = scale(U, i, self.maxNorm/normUi, self.k)             
            
            if ind > self.startAverage: 
                muU[i, :] = muU[i, :]*ind/float(ind+1) + U[i, :]/float(ind+1)
            else: 
                muU[i, :] = U[i, :]
                
            #Now update V
            #r = SparseUtilsCython.computeR(U, V, w, numAucSamples)        
            j = permutedColInds[s % permutedColInds.shape[0]]
            dVj = self.derivativeViApprox(indPtr, colInds, U, V, gp, gq, normGp, normGq, permutedRowInds, permutedColInds, j)
            plusEquals(V, j, -sigma*dVj, self.k)
            normVj = numpy.linalg.norm(V[j,:])  
            
            if normVj >= self.maxNorm: 
                V[j,:] = scale(V, j, self.maxNorm/normVj, self.k)        
            
            if ind > self.startAverage: 
                muV[j, :] = muV[j, :]*ind/float(ind+1) + V[j, :]/float(ind+1)
            else: 
                muV[j, :] = V[j, :]
    
    
    def updateUVApprox2(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=2, mode="c"] muU, numpy.ndarray[double, ndim=2, mode="c"] muV, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedRowInds,  numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[double, ndim=1, mode="c"] normGp, numpy.ndarray[double, ndim=1, mode="c"] normGq, unsigned int ind, unsigned int numIterations, double sigma): 
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0]    
        cdef unsigned int i, j, s
        cdef double normUi, normVj
        cdef bint newline = indPtr.shape[0] > 100000
        cdef numpy.ndarray[double, ndim=1, mode="c"] dUi = numpy.zeros(self.k)
        cdef numpy.ndarray[double, ndim=1, mode="c"] dVj = numpy.zeros(self.k)
    
        for s in range(numIterations):
            if s % self.printStep == 0: 
                if newline:  
                    print(str(s) + " of " + str(numIterations))
                else: 
                    print(str(s) + " ", end="")
                     
            i = permutedRowInds[s % permutedRowInds.shape[0]]   
            
            if self.itemFactors: 
                U[i,:] = self.meanPositive(indPtr, colInds, V, i)
            else: 
                dUi = self.derivativeUiApprox2(indPtr, colInds, U, V, gp, gq, permutedColInds, i)
                plusEquals(U, i, -sigma*dUi, self.k)
                normUi = numpy.linalg.norm(U[i,:])
                
                if normUi >= self.maxNorm: 
                    U[i,:] = scale(U, i, self.maxNorm/normUi, self.k)             
            
            if ind > self.startAverage: 
                muU[i, :] = muU[i, :]*ind/float(ind+1) + U[i, :]/float(ind+1)
            else: 
                muU[i, :] = U[i, :]
                
            #Now update V
            #r = SparseUtilsCython.computeR(U, V, w, numAucSamples)        
            j = permutedColInds[s % permutedColInds.shape[0]]
            dVj = self.derivativeViApprox2(indPtr, colInds, U, V, gp, gq, normGp, normGq, permutedRowInds, permutedColInds, j)
            plusEquals(V, j, -sigma*dVj, self.k)
            normVj = numpy.linalg.norm(V[j,:])  
            
            if normVj >= self.maxNorm: 
                V[j,:] = scale(V, j, self.maxNorm/normVj, self.k)        
            
            if ind > self.startAverage: 
                muV[j, :] = muV[j, :]*ind/float(ind+1) + V[j, :]/float(ind+1)
            else: 
                muV[j, :] = V[j, :]

          
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
                    partialObj += gp[p]*tanh(kappa/normGq)
               
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
                    partialObj += gp[p]*tanh(kappa/normGq)
               
            if normGp != 0: 
                objVector[i] = partialObj/normGp
        
        objVector /= 2*m  
        objVector += (0.5/m)*((self.lmbdaV/m)*numpy.linalg.norm(V)**2 + (self.lmbdaU/m)*numpy.linalg.norm(U)**2) 
        
        if full: 
            return objVector
        else: 
            return objVector.sum() 
