#cython: profile=True 
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


cdef class MaxLocalAUCCython(object): 
    cdef public unsigned int k, printStep, numAucSamples, numRowSamples, startAverage
    cdef public double lmbdaU, lmbdaV, maxNorm, rho, w
    cdef public bint itemFactors, normalise
    
    def __init__(self, k=8, lmbdaU=0.0, lmbdaV=1.0, normalise=True, numAucSamples=10, numRowSamples=30, startAverage=30, rho=0.5, w=0.9): 
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
    
    def derivativeUi(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r,  numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, unsigned int i):
        """
        Find  delta phi/delta u_i using the hinge loss.  
        """
        cdef unsigned int p, q
        cdef double uivp, uivq, gamma, kappa, ri
        cdef double  normDeltaTheta, hGamma, hKappa, vpScale, normGp, normGq 
        cdef unsigned int m = U.shape[0], n = V.shape[0], numOmegai, numOmegaBari
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaBari 
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(self.k, numpy.float)
          
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint32), omegai, assume_unique=True)
        numOmegai = omegai.shape[0]
        numOmegaBari = n-numOmegai
        
        deltaTheta = numpy.zeros(self.k)
        ri = r[i]
        normGp = gp[omegai].sum()
        normGq = gq[omegaBari].sum()
        
        if numOmegai * numOmegaBari != 0:         
            for p in omegai: 
                vpScale = 0
                uivp = dot(U, i, V, p, self.k)
                kappa = self.rho*(uivp - ri)
                hKappa = 1-kappa
                
                for q in omegaBari: 
                    uivq = dot(U, i, V, q, self.k)
                    
                    gamma = uivp - uivq
                    hGamma = 1-gamma 
                    
                    if hGamma > 0 and hKappa > 0: 
                        #vpScale -= hGamma*(hKappa**2) + (hGamma**2)*hKappa*rho
                        #deltaTheta += V[q, :]*hGamma*(hKappa**2)
                        deltaTheta += gp[p] * gq[q] * ((V[q, :] - V[p, :])*hGamma*tanh(hKappa) - V[p, :]*(self.rho/2)*(hGamma**2)*(1 - tanh(hKappa)**2))/(normGp*normGq)
                    
            deltaTheta *= gi[i]
                    
        #Normalise gradient to have unit norm 
        normDeltaTheta = numpy.linalg.norm(deltaTheta)
        
        if normDeltaTheta != 0 and self.normalise: 
            deltaTheta = deltaTheta/normDeltaTheta
        
        return deltaTheta
    
    def updateU(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r, numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, double sigma):  
        """
        Compute the full gradient descent update of U
        """    
        
        cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dU = numpy.zeros((U.shape[0], U.shape[1]), numpy.float)
        cdef unsigned int i 
        cdef unsigned int m = U.shape[0]
        
        for i in range(m): 
            dU[i, :] = self.derivativeUi(indPtr, colInds, U, V, r, gi, gp, gq, i) 
        
        U -= sigma*dU
        
        for i in range(m):
            U[i,:] = scale(U, i, 1/numpy.linalg.norm(U[i,:]), self.k)   
    
    
    def derivativeUiApprox(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V,  numpy.ndarray[double, ndim=1, mode="c"] r,  numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, unsigned int i):
        """
        Find an approximation of delta phi/delta u_i using the simple objective without 
        sigmoid functions. 
        """
        cdef unsigned int p, q, ind, j, s
        cdef double uivp, ri, uivq, gamma, kappa, hGamma, hKappa
        cdef double normDeltaTheta, vqScale, vpScale, normGp=0, normGpq=0, tanhHKappa, rhoOver2 = self.rho/2
        cdef unsigned int m = U.shape[0], n = V.shape[0], numOmegai, numOmegaBari
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaiSample
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(self.k, numpy.float)
        cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ = numpy.zeros(self.k, numpy.int)
             
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        numOmegai = omegai.shape[0]
        numOmegaBari = n-numOmegai
        ri = r[i]
        
        deltaTheta = numpy.zeros(self.k)
        
        if numOmegai * numOmegaBari != 0: 
            omegaiSample = uniformChoice(omegai, self.numAucSamples)   
            
            for p in omegaiSample:
                q = inverseChoiceArray(omegai, permutedColInds) 
            
                uivp = dot(U, i, V, p, self.k)
                uivq = dot(U, i, V, q, self.k)
                
                gamma = uivp - uivq
                kappa = self.rho*(uivp - ri) 
                hGamma = max(1 - gamma, 0)
                hKappa = max(1 - kappa, 0)
                
                zeta = gp[p]*gq[q]
                normGpq += zeta
                tanhHKappa = tanh(hKappa)           
                
                if hGamma > 0 and hKappa > 0: 
                    vqScale = zeta*hGamma*tanhHKappa
                    vpScale = -vqScale - zeta*rhoOver2*square(hGamma)*(1 - square(tanhHKappa))
                    
                    deltaTheta += scale(V, p, vpScale, self.k) + scale(V, q, vqScale, self.k)    
                    
            #if normGp*normGq != 0: 
            deltaTheta *= gi[i]/normGpq
            
        deltaTheta += scale(U, i, self.lmbdaU/m, self.k)
                        
        #Normalise gradient to have unit norm 
        normDeltaTheta = numpy.linalg.norm(deltaTheta)
        
        if normDeltaTheta != 0 and self.normalise: 
            deltaTheta = deltaTheta/normDeltaTheta
        
        return deltaTheta
    
    
    def derivativeVi(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r, numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, unsigned int j): 
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
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaBari
        
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.uint32), omegai, assume_unique=True)
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
                    kappa = self.rho*(uivp - ri)
                    
                    hGamma = 1-gamma 
                    hKappa = 1-kappa
                    
                    normGq += gq[q]
    
                    if hGamma > 0 and hKappa>0: 
                        #betaScale += hGamma*hKappa**2 + hGamma**2*hKappa*self.rho
                        betaScale += gp[p] * gq[q] * (hGamma*tanh(hKappa) + (self.rho/2)*hGamma**2 * (1- tanh(hKappa)**2))
                
                if normGp*normGq != 0: 
                    deltaBeta = scale(U, i, -betaScale/(normGp*normGq), k)
            else:
                q = j 
                uivq = dot(U, i, V, q, k)
                
                normGq = gq[omegaBari].sum()
                                
                for p in omegai: 
                    uivp = dot(U, i, V, p, k)
                    gamma = uivp - uivq  
                    kappa = self.rho*(uivp - ri)
                    
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
        deltaTheta += scale(V, j, self.lmbdaV/n, self.k)
        
        #Make gradient unit norm 
        normTheta = numpy.linalg.norm(deltaTheta)
        if normTheta != 0 and self.normalise: 
            deltaTheta = deltaTheta/normTheta
        
        return deltaTheta
     
    
    def updateV(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r, numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, double sigma): 
        """
        Compute the full gradient descent update of V
        """
        cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dV = numpy.zeros((V.shape[0], V.shape[1]), numpy.float)
        cdef unsigned int j
        cdef unsigned int n = V.shape[0]
        cdef unsigned int k = V.shape[1]
        
        for j in range(n): 
            dV[j, :] = self.derivativeVi(indPtr, colInds, U, V, r, gi, gp, gq, j) 
            
        V -= sigma*dV
        
        for j in range(n): 
            normVj = numpy.linalg.norm(V[j,:])        
            if normVj >= self.lmbdaV: 
                V[j,:] = scale(V, j, self.lmbdaV/normVj, k)               
    
    def derivativeViApprox(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r,  numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[double, ndim=1, mode="c"] normGp, numpy.ndarray[double, ndim=1, mode="c"] normGq, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedRowInds,  numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, unsigned int j): 
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
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(self.k, numpy.float)
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(self.k, numpy.float)
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] rowInds = numpy.random.choice(permutedRowInds, min(self.numRowSamples, permutedRowInds.shape[0]), replace=False)
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaiSample
        
        rhoOver2 = self.rho/2    
        
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
                uivp = dot(U, i, V, p, self.k)
                nu = 1 - uivp
                hKappa = max(0, 1 - self.rho*(uivp - ri))
                zeta = tanh(hKappa)
                #normGp = partialSum(gp, omegai)
    
                for s in range(self.numAucSamples): 
                    q = inverseChoiceArray(omegai, permutedColInds)
                    uivq = dot(U, i, V, q, self.k)                
                    hGamma = max(0, nu+uivq) 
                    normGqi += gq[q]
                    
                    betaScale += gp[p] * gq[q] * (hGamma*zeta + rhoOver2*square(hGamma) * (1 - square(zeta)))  
                                 
                if normGp[i]*normGqi != 0:
                    deltaBeta = scale(U, i, -betaScale/(normGp[i]*normGqi), self.k)
            elif numOmegai != 0:
                q = j 
                uivq = dot(U, i, V, q, self.k)
                nu = 1 + uivq 
                nuPrime = 1 + ri*self.rho
                omegaiSample = uniformChoice(omegai, self.numAucSamples)
    
                for p in omegaiSample: 
                    uivp = dot(U, i, V, p, self.k)
                    hGamma = max(0, nu - uivp) 
                    hKappa = max(0, nuPrime - self.rho*uivp)
                    normGpi += gp[p]
                    
                    betaScale += gp[p] * gq[q]*hGamma*tanh(hKappa)
                
                if normGpi*normGq[i] != 0:
                    deltaBeta = scale(U, i, betaScale/(normGpi*normGq[i]), self.k)  
                    
            deltaTheta += deltaBeta*gi[i]
            
        deltaTheta /= gi[rowInds].sum()
        deltaTheta += scale(V, j, self.lmbdaV/n, self.k)
        
        #Make gradient unit norm 
        normTheta = numpy.linalg.norm(deltaTheta)
        if normTheta != 0 and self.normalise: 
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
        
    
    def updateUVApprox(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=2, mode="c"] muU, numpy.ndarray[double, ndim=2, mode="c"] muV, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedRowInds,  numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, numpy.ndarray[double, ndim=1, mode="c"] normGp, numpy.ndarray[double, ndim=1, mode="c"] normGq, unsigned int ind, unsigned int numIterations, double sigma): 
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0]    
        cdef unsigned int i, j, s
        cdef double normUi, normVj
        cdef numpy.ndarray[double, ndim=1, mode="c"] dUi = numpy.zeros(self.k)
        cdef numpy.ndarray[double, ndim=1, mode="c"] dVj = numpy.zeros(self.k)
        cdef numpy.ndarray[double, ndim=1, mode="c"] r 
    
        for s in range(numIterations):
            if s % self.printStep == 0: 
                print(str(s) + " ", end="")
                
            r = SparseUtilsCython.computeR(U, V, self.w, self.numAucSamples)
            
            i = permutedRowInds[s % permutedRowInds.shape[0]]   
            
            if self.itemFactors: 
                U[i,:] = self.meanPositive(indPtr, colInds, V, i)
            else: 
                dUi = self.derivativeUiApprox(indPtr, colInds, U, V, r, gi, gp, gq, permutedColInds, i)
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
            dVj = self.derivativeViApprox(indPtr, colInds, U, V, r, gi, gp, gq, normGp, normGq, permutedRowInds, permutedColInds, j)
            plusEquals(V, j, -sigma*dVj, self.k)
            normVj = numpy.linalg.norm(V[j,:])  
            
            if normVj >= self.maxNorm: 
                V[j,:] = scale(V, j, self.maxNorm/normVj, self.k)        
            
            if ind > self.startAverage: 
                muV[j, :] = muV[j, :]*ind/float(ind+1) + V[j, :]/float(ind+1)
            else: 
                muV[j, :] = V[j, :]
               
    def objectiveApprox(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[unsigned int, ndim=1, mode="c"] allIndPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] allColInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r,   numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, bint full=False):         
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0]
        cdef unsigned int i, j, k, p, q
        cdef double uivp, uivq, gamma, kappa, ri, partialObj, hGamma, hKappa, normGp, normGq, normGi=0, zeta, normGpq
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] allOmegai 
        cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] omegaiSample
        cdef numpy.ndarray[double, ndim=1, mode="c"] objVector = numpy.zeros(m, dtype=numpy.float)
    
        k = U.shape[1]
        
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            allOmegai = allColInds[allIndPtr[i]:allIndPtr[i+1]]
            
            ri = r[i]
            normGpq = 0 
            normGi += gi[i]
            partialObj = 0                
            
            omegaiSample = uniformChoice(omegai, self.numAucSamples) 
            #omegaiSample = omegai
            
            for p in omegaiSample:
                q = inverseChoice(allOmegai, n)                  
            
                uivp = dot(U, i, V, p, k)
                uivq = dot(U, i, V, q, k)
                
                gamma = uivp - uivq
                hGamma = max(0, 1 - gamma)
                                
                kappa = self.rho*(uivp - ri)
                hKappa = max(0, 1 - kappa)
                
                zeta = gp[p]*gq[q]
                normGpq += zeta
                
                partialObj += zeta * square(hGamma) * tanh(hKappa)
            
            if normGpq != 0: 
                objVector[i] = partialObj*gi[i]/normGpq
        
        objVector /= 2*normGi
        objVector += 0.5*(self.lmbdaV/(n*m))*numpy.linalg.norm(V)**2
        
        if full: 
            return objVector
        else: 
            return objVector.sum() 
      
    def objective(self, numpy.ndarray[unsigned int, ndim=1, mode="c"] indPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] colInds, numpy.ndarray[unsigned int, ndim=1, mode="c"] allIndPtr, numpy.ndarray[unsigned int, ndim=1, mode="c"] allColInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] r,  numpy.ndarray[double, ndim=1, mode="c"] gi, numpy.ndarray[double, ndim=1, mode="c"] gp, numpy.ndarray[double, ndim=1, mode="c"] gq, bint full=False):         
        """
        Note that distributions gp, gq and gi must be normalised to have sum 1. 
        """
        cdef unsigned int m = U.shape[0]
        cdef unsigned int n = V.shape[0]
        cdef unsigned int i, j, p, q
        cdef double uivp, uivq, gamma, kappa, ri, hGamma, hKappa, normGpq, normGi=gi.sum(), sumQ=0
        cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
        cdef numpy.ndarray[int, ndim=1, mode="c"] omegaBari 
        cdef numpy.ndarray[int, ndim=1, mode="c"] allOmegai 
        cdef numpy.ndarray[double, ndim=1, mode="c"] objVector = numpy.zeros(m, dtype=numpy.float)
    
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            allOmegai = allColInds[allIndPtr[i]:allIndPtr[i+1]]
            ri = r[i]
            
            omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.int32), omegai, assume_unique=True)
            partialObj = 0 
            normGpq = 0
            
            for p in omegai:
                uivp = dot(U, i, V, p, self.k)
                
                kappa = self.rho*(uivp - ri)
                hKappa = max(0, 1-kappa)

                for q in omegaBari:                 
                    uivq = dot(U, i, V, q, self.k)
                    gamma = uivp - uivq
                    hGamma = max(0, 1-gamma)
                    
                    normGpq += gp[p]*gq[q]
                    partialObj += gp[p]*gq[q]*hGamma**2 * tanh(hKappa)
                
            objVector[i] = gi[i]*partialObj/normGpq
        
        objVector /= 2*normGi  
        objVector += 0.5*(self.lmbdaV/(n*m))*numpy.linalg.norm(V)**2
        
        if full: 
            return objVector
        else: 
            return objVector.sum() 
