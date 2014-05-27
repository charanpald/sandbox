#cython: profile=True 
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
import cython
from cython.parallel import parallel, prange
cimport numpy
import numpy
from sandbox.util.CythonUtils cimport dot, scale, choice, inverseChoice, uniformChoice, plusEquals


from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX

cdef extern from "math.h":
    double exp(double x)
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

def derivativeUi(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] xi, unsigned int i, double lmbda, double C, bint normalise):
    """
    Find  delta phi/delta u_i using the hinge loss.  
    """
    cdef unsigned int p, q
    cdef unsigned int k = U.shape[1]
    cdef double uivp, uivq, gamma
    cdef double  normDeltaTheta, zeta 
    cdef unsigned int m = U.shape[0], n = V.shape[0], numOmegai, numOmegaBari
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaBari 
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
      
    omegai = colInds[indPtr[i]:indPtr[i+1]]
    omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.int32), omegai, assume_unique=True)
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

def updateU(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] xi, double sigma, double lmbda, double C, bint normalise):  
    """
    Compute the full gradient descent update of U
    """    
    
    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dU = numpy.zeros((U.shape[0], U.shape[1]), numpy.float)
    cdef unsigned int i 
    cdef unsigned int m = U.shape[0]
    cdef unsigned int k = U.shape[1]
    
    for i in range(m): 
        dU[i, :] = derivativeUi(indPtr, colInds, U, V, xi, i, lmbda, C, normalise) 
    
    U -= sigma*dU
    
    for i in range(m):
        U[i,:] = scale(U, i, 1/numpy.linalg.norm(U[i,:]), k)   

def derivativeUiApprox(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=1, mode="c"] colIndsCumProbs, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V,  numpy.ndarray[double, ndim=1, mode="c"] xi, unsigned int i, unsigned int numRowSamples, unsigned int numAucSamples, double lmbda, double C, bint normalise):
    """
    Find an approximation of delta phi/delta u_i using the simple objective without 
    sigmoid functions. 
    """
    cdef unsigned int p, q, ind, j, s
    cdef unsigned int k = U.shape[1]
    cdef double uivp, ri, uivq, gamma
    cdef double normDeltaTheta, vqScale, vpScale, zeta  
    cdef unsigned int m = U.shape[0], n = V.shape[0], numOmegai, numOmegaBari
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaiSample
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] omegaProbsi = numpy.zeros(k)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] indsQ = numpy.zeros(k, numpy.int)
         
    omegai = colInds[indPtr[i]:indPtr[i+1]]
    omegaProbsi = colIndsCumProbs[indPtr[i]:indPtr[i+1]]
    numOmegai = omegai.shape[0]
    numOmegaBari = n-numOmegai
    
    deltaTheta = numpy.zeros(k)
    
    if numOmegai * numOmegaBari != 0: 
        omegaiSample = choice(omegai, numAucSamples, omegaProbsi)   
        #omegaiSample = choice(omegai, numAucSamples, computeOmegaProbs(i, omegai, U, V))
        #omegaiSample = numpy.random.choice(omegai, numAucSamples, p=numpy.r_[omegaProbsi[0], numpy.diff(omegaProbsi)])
        
        for p in omegaiSample:
            q = inverseChoice(omegai, n) 
        
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

def derivativeVi(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] xi, unsigned int j, double lmbda, double C, bint normalise): 
    """
    delta phi/delta v_i using hinge loss. 
    """
    cdef unsigned int i = 0
    cdef unsigned int k = U.shape[1]
    cdef unsigned int p, q, numOmegai, numOmegaBari, t
    cdef unsigned int m = U.shape[0]
    cdef unsigned int n = V.shape[0], ind
    cdef unsigned int s = 0
    cdef double uivp, uivq,  betaScale, ri, normTheta
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaBari
    
    for i in range(m): 
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.int32), omegai, assume_unique=True)
        numOmegai = omegai.shape[0]       
        numOmegaBari = n-numOmegai
        
        betaScale = 0
        
        if j in omegai:                 
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
 

def updateV(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] xi, double sigma, double lmbda, double C, bint normalise): 
    """
    Compute the full gradient descent update of V
    """
    cdef numpy.ndarray[numpy.float_t, ndim=2, mode="c"] dV = numpy.zeros((V.shape[0], V.shape[1]), numpy.float)
    cdef unsigned int j
    cdef unsigned int n = V.shape[0]
    cdef unsigned int k = V.shape[1]
    
    for j in range(n): 
        dV[j, :] = derivativeVi(indPtr, colInds, U, V, xi, j, lmbda, C, normalise) 
    
    V -= sigma*dV

def derivativeViApprox(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=1, mode="c"] colIndsCumProbs, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] xi,  unsigned int j, unsigned int numRowSamples, unsigned int numAucSamples, double lmbda, double C, bint normalise): 
    """
    delta phi/delta v_i  using the hinge loss. 
    """
    cdef unsigned int i = 0
    cdef unsigned int k = U.shape[1]
    cdef unsigned int p, q, numOmegai, numOmegaBari
    cdef unsigned int m = U.shape[0]
    cdef unsigned int n = V.shape[0]
    cdef unsigned int s = 0
    cdef double uivp, uivq,  betaScale, xii, normTheta, zeta, gamma, nu
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaBeta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] deltaTheta = numpy.zeros(k, numpy.float)
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] omegaProbsi
    cdef numpy.ndarray[numpy.int_t, ndim=1, mode="c"] rowInds = numpy.random.permutation(m)[0:numRowSamples]
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaiSample
    
    for i in rowInds: 
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        omegaProbsi = colIndsCumProbs[indPtr[i]:indPtr[i+1]]
        numOmegai = omegai.shape[0]       
        numOmegaBari = n-numOmegai
        
        betaScale = 0
        xii = xi[i]
        
        if j in omegai:                 
            p = j 
            uivp = dot(U, i, V, p, k)
            nu = 1 - uivp - xii

            for s in range(numAucSamples): 
                q = inverseChoice(omegai, n)
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
            omegaiSample = choice(omegai, numAucSamples, omegaProbsi)
            #omegaiSample = choice(omegai, numAucSamples, computeOmegaProbs(i, omegai, U, V))
            #omegaiSample = numpy.random.choice(omegai, numAucSamples, p=numpy.r_[omegaProbsi[0], numpy.diff(omegaProbsi)])

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

def derivativeXii(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] xi, unsigned int i, double lmbda, double C, bint normalise):
    """
    Find  delta phi/delta u_i using the hinge loss.  
    """
    cdef unsigned int p, q, j
    cdef unsigned int k = U.shape[1]
    cdef double uivp, uivq, gamma, zeta, deltaTheta
    cdef unsigned int m = U.shape[0], n = V.shape[0], numOmegai, numOmegaBari
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaBari
      
    omegai = colInds[indPtr[i]:indPtr[i+1]]
    omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.int32), omegai, assume_unique=True)
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
                    deltaTheta -= zeta
                
        deltaTheta /= float(numOmegai * numOmegaBari * m)
    
    deltaTheta += C/m
    
    return deltaTheta

def derivativeXiiApprox(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=1, mode="c"] colIndsCumProbs, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] xi, unsigned int i, unsigned int numRowSamples, unsigned int numAucSamples,  double lmbda, double C, bint normalise):
    """
    Find an approximation of delta phi/delta u_i using the simple objective without 
    sigmoid functions. 
    """
    cdef unsigned int p, q
    cdef unsigned int k = U.shape[1]    
    cdef unsigned int m = U.shape[0], n = V.shape[0], numOmegai, numOmegaBari
    cdef double uivp, uivq, gamma, zeta, deltaTheta 
    cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] omegaProbsi
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaiSample
                
    omegai = colInds[indPtr[i]:indPtr[i+1]]
    omegaProbsi = colIndsCumProbs[indPtr[i]:indPtr[i+1]]
    numOmegai = omegai.shape[0]
    numOmegaBari = n-numOmegai
    
    deltaTheta = 0
    
    if numOmegai * numOmegaBari != 0: 
        omegaiSample = choice(omegai, numAucSamples, omegaProbsi) 
        #omegaiSample = choice(omegai, numAucSamples, computeOmegaProbs(i, omegai, U, V))
        #omegaiSample = numpy.random.choice(omegai, numAucSamples, p=numpy.r_[omegaProbsi[0], numpy.diff(omegaProbsi)])
        
        for p in omegaiSample:
            q = inverseChoice(omegai, n) 
        
            uivp = dot(U, i, V, p, k)
            uivq = dot(U, i, V, q, k)
            
            gamma = uivp - uivq
            zeta = 1 - gamma - xi[i]
            
            if zeta > 0: 
                deltaTheta -= zeta 
            
        deltaTheta /= float(omegaiSample.shape[0] * m)
    
    deltaTheta += C/m 

    return deltaTheta

def updateUVApprox(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=2, mode="c"] muU, numpy.ndarray[double, ndim=2, mode="c"] muV, numpy.ndarray[double, ndim=1, mode="c"] xi, numpy.ndarray[double, ndim=1, mode="c"] muXi, numpy.ndarray[double, ndim=1, mode="c"] colIndsCumProbs, numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedRowInds,  numpy.ndarray[unsigned int, ndim=1, mode="c"] permutedColInds, unsigned int ind, double sigma, unsigned int numRowSamples, unsigned int numAucSamples, double w, double lmbda, double C, bint normalise): 
    cdef unsigned int m = U.shape[0]
    cdef unsigned int n = V.shape[0]    
    cdef unsigned int k = U.shape[1] 
    cdef double normUi
    cdef numpy.ndarray[double, ndim=1, mode="c"] dUi = numpy.zeros(k)
    cdef numpy.ndarray[double, ndim=1, mode="c"] dVj = numpy.zeros(k)
    cdef unsigned int i, j, s, ind2, dXii
    cdef unsigned int startAverage = 10
    
    for s in range(m):
        i = permutedRowInds[(ind + s) % m]
        dUi = derivativeUiApprox(indPtr, colInds, colIndsCumProbs, U, V, xi, i, numRowSamples, numAucSamples, lmbda, C, normalise)
        
        j = permutedColInds[(ind + s) % n]
        dVj = derivativeViApprox(indPtr, colInds, colIndsCumProbs, U, V, xi, j, numRowSamples, numAucSamples, lmbda, C, normalise)

        dXii = derivativeXiiApprox(indPtr, colInds, colIndsCumProbs, U, V, xi, i, numRowSamples, numAucSamples, lmbda, C, normalise)

        plusEquals(U, i, -sigma*dUi, k)
        
        normUi = numpy.linalg.norm(U[i,:])
        
        if normUi != 0: 
            U[i,:] = scale(U, i, 1/normUi, k)             
        
        plusEquals(V, j, -sigma*dVj, k)
                
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
         
def objectiveApprox(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[int, ndim=1, mode="c"] allIndPtr, numpy.ndarray[int, ndim=1, mode="c"] allColInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] xi, unsigned int numAucSamples, double lmbda, double C, bint full=False):         
    cdef unsigned int m = U.shape[0]
    cdef unsigned int n = V.shape[0]
    cdef unsigned int i, j, k, p, q
    cdef double uivp, uivq, gamma, zeta
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] allOmegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaiSample
    cdef numpy.ndarray[double, ndim=1, mode="c"] objVector = numpy.zeros(m, dtype=numpy.float)

    k = U.shape[1]
    
    for i in range(m): 
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        allOmegai = allColInds[allIndPtr[i]:allIndPtr[i+1]]

        if omegai.shape[0] * (n-omegai.shape[0]) != 0: 
            partialObj = 0                
            
            omegaiSample = uniformChoice(omegai, numAucSamples) 
            #omegaiSample = omegai
            
            for p in omegaiSample:
                q = inverseChoice(allOmegai, n)                  
            
                uivp = dot(U, i, V, p, k)
                gamma = uivp - uivq
                
                uivq = dot(U, i, V, q, k)
                zeta = 1-gamma-xi[i]
                
                if zeta > 0: 
                    partialObj += zeta**2
                
            objVector[i] = partialObj/float(omegaiSample.shape[0])
    
    objVector /= 2       
    objVector += C*xi + (lmbda/(2*m))*numpy.linalg.norm(V)**2
    
    if full: 
        return objVector
    else: 
        return objVector.mean() 
  
def objective(numpy.ndarray[int, ndim=1, mode="c"] indPtr, numpy.ndarray[int, ndim=1, mode="c"] colInds, numpy.ndarray[int, ndim=1, mode="c"] allIndPtr, numpy.ndarray[int, ndim=1, mode="c"] allColInds, numpy.ndarray[double, ndim=2, mode="c"] U, numpy.ndarray[double, ndim=2, mode="c"] V, numpy.ndarray[double, ndim=1, mode="c"] xi, double lmbda, double C, bint full=False):         
    cdef unsigned int m = U.shape[0]
    cdef unsigned int n = V.shape[0]
    cdef unsigned int i, j, k, p, q
    cdef double uivp, uivq, gamma, zeta
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegai 
    cdef numpy.ndarray[int, ndim=1, mode="c"] omegaBari 
    cdef numpy.ndarray[int, ndim=1, mode="c"] allOmegai 
    cdef numpy.ndarray[double, ndim=1, mode="c"] objVector = numpy.zeros(m, dtype=numpy.float)

    k = U.shape[1]
    
    for i in range(m): 
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        allOmegai = allColInds[allIndPtr[i]:allIndPtr[i+1]]
        
        omegaBari = numpy.setdiff1d(numpy.arange(n, dtype=numpy.int32), omegai, assume_unique=True)

        if omegai.shape[0] * (n-omegai.shape[0]) != 0: 
            partialObj = 0                
                        
            for p in omegai:
                for q in omegaBari:                 
                    uivp = dot(U, i, V, p, k)
                    gamma = uivp - uivq
                    
                    uivq = dot(U, i, V, q, k)
                    zeta = 1-gamma-xi[i]
                    
                    if zeta > 0: 
                        partialObj += zeta**2
                
            objVector[i] = partialObj/float(omegai.shape[0]*omegaBari.shape[0])
    
    objVector /= 2       
    objVector += C*xi + (lmbda/(2*m))*numpy.linalg.norm(V)**2
    
    if full: 
        return objVector
    else: 
        return objVector.mean() 
