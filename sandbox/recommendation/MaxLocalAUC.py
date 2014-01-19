
import numpy 
import logging
from math import exp
from sandbox.util.SparseUtils import SparseUtils
from sandbox.recommendation.MaxLocalAUCCython import derivativeUi, derivativeVi, derivativeUiApprox, derivativeViApprox


class MaxLocalAUC(object): 
    """
    Let's try different ways of maximising the local AUC with a penalty term. 
    """
    def __init__(self, lmbda, k, r, sigma=0.05, eps=0.1, stochastic=False): 
        self.lmbda = lmbda
        self.k = k 
        self.r = r
        self.sigma = sigma
        self.eps = eps 
        self.stochastic = stochastic
        
        self.recordStep = 5
        self.stocrecordStep = 10

        self.pythonDerivative = False 
        self.approxDerivative = True 
        
        self.numSamples = 200
        self.maxIterations = 20
        
        self.initialAlg = "rand"
    
    def getOmegaList(self, X): 
        """
        Return a list such that the ith element contains an array of nonzero 
        entries in the ith row of X. 
        """
        omegaList = []
        
        for i in range(X.shape[0]): 
            omegaList.append(numpy.array(X[i, :].nonzero()[1], numpy.uint))
        return omegaList 
    
    def learnModel(self, X, verbose=False): 
        """
        Max local AUC with Frobenius norm penalty. Solve with gradient descent. 
        The input is a sppy.csarray object 
        """
        
        m = X.shape[0]
        n = X.shape[1]
        omegaList = self.getOmegaList(X)

        rowInds, colInds = X.nonzero()
        mStar = numpy.unique(rowInds).shape[0]
        
        if self.initialAlg == "rand": 
            U = numpy.random.rand(m, self.k)
            V = numpy.random.rand(n, self.k)
        elif self.initialAlg == "zeros": 
            U = numpy.zeros((m, self.k))
            V = numpy.zeros((n, self.k))
        elif self.initialAlg == "svd":
            logging.debug("Initialising with SVD")
            U, s, V = SparseUtils.svdPropack(X, self.k)
            U = numpy.ascontiguousarray(U)
            V = numpy.ascontiguousarray(V)
        else:
            raise ValueError("Unknown initialisation: " + str(self.initialAlg))
        
        lastU = U+numpy.ones((m, self.k))*self.eps
        lastV = V+numpy.ones((n, self.k))*self.eps 
        
        normDeltaU = numpy.linalg.norm(U - lastU)
        normDeltaV = numpy.linalg.norm(V - lastV)
        objs = []
        aucs = []
        
        ind = 0
        
        eps = self.eps 
        
        while (normDeltaU > eps or normDeltaV > eps) and ind < self.maxIterations: 
            lastU = U.copy() 
            lastV = V.copy() 
            
            deltaU = self.sigma*self.derivativeU(X, U, V, omegaList, mStar)
            deltaV = self.sigma*self.derivativeV(X, U, V, omegaList)
            
            U -= deltaU
            V -= deltaV

            normDeltaU = numpy.linalg.norm(U - lastU)
            normDeltaV = numpy.linalg.norm(V - lastV)               
                
            if (not self.stochastic and ind % self.recordStep == 0) or (self.stochastic and ind % self.stocrecordStep == 0): 
                objs.append(self.objectiveApprox(X, U, V, omegaList)) 
                aucs.append(self.localAUCApprox(X, U, V, omegaList))
                printStr = "Iteration: " + str(ind)
                printStr += " local AUC~" + str(aucs[-1]) + " objective~" + str(objs[-1])
                printStr += " ||dU||=" + str(normDeltaU) + " " + "||dV||=" + str(normDeltaV)
                logging.debug(printStr)
            
            ind += 1
                        
        if verbose:     
            return U, V, numpy.array(objs), numpy.array(aucs), ind
        else: 
            return U, V
        
    def derivativeU(self, X, U, V, omegaList, mStar): 
        """
        Find the derivative for all of U. 
        """
        dU = numpy.zeros(U.shape)
        
        if not self.stochastic: 
            inds = numpy.arange(X.shape[0])
        else: 
            inds = numpy.random.randint(0, X.shape[0], self.numSamples)
        
        if self.pythonDerivative: 
            for i in inds: 
                dU[i, :] = self.derivativeUi(X, U, V, omegaList, i, mStar)
        else:
            if self.approxDerivative: 
                for i in inds: 
                    dU[i, :] = derivativeUiApprox(X, U, V, omegaList, i, mStar, self.numSamples, self.k, self.lmbda, self.r)
            else:    
                for i in inds:
                    dU[i, :] = derivativeUi(X, U, V, omegaList, i, mStar, self.k, self.lmbda, self.r)
            
        return dU 
        
    #@profile
    def derivativeUi(self, X, U, V, omegaList, i, mStar): 
        """
        delta phi/delta u_i
        """
        deltaPhi = self.lmbda * U[i, :]
        
        omegai = omegaList[i]
        omegaBari = numpy.setdiff1d(numpy.arange(X.shape[1]), omegai)
        
        deltaAlpha = numpy.zeros(self.k)
        
        ui = U[i, :]
        
        for p in omegai: 
            vp = V[p, :]
            uivp = ui.dot(vp)
            kappa = exp(-uivp +self.r[i])
            onePlusKappa = 1+kappa
            onePlusKappSq = onePlusKappa**2
            
            for q in omegaBari: 
                vq = V[q, :]
                uivq = ui.dot(vq)
                gamma = exp(-uivp+uivq)
                onePlusGamma = 1+gamma
                
                denom = onePlusGamma**2 * onePlusKappSq
                deltaAlpha += (vp*gamma*onePlusKappa + vq*(kappa-gamma))/denom 
                
        if omegai.shape[0] * omegaBari.shape[0] != 0: 
            deltaAlpha /= float(omegai.shape[0] * omegaBari.shape[0]*mStar)
            
        deltaPhi -= deltaAlpha
        
        return deltaPhi
        
    def derivativeV(self, X, U, V, omegaList): 
        """
        Find the derivative for all of V. 
        """
        dV = numpy.zeros(V.shape)
        X = numpy.array(X.todense())
        
        if not self.stochastic: 
            inds = numpy.arange(X.shape[1])
        else: 
            inds = numpy.random.randint(0, X.shape[1], self.numSamples)
        
        if self.pythonDerivative: 
            for i in inds: 
                V[i, :] = self.derivativeVi(X, U, V, omegaList, i)
        else: 
            if self.approxDerivative: 
                for i in inds: 
                    dV[i, :] = derivativeViApprox(X, U, V, omegaList, i, self.numSamples, self.k, self.lmbda, self.r)
            else: 
                for i in inds: 
                    dV[i, :] = derivativeVi(X, U, V, omegaList, i, self.k, self.lmbda, self.r)
            
        return dV    
      
      
    #@profile    
    def derivativeVi(self, X, U, V, omegaList, j): 
        """
        delta phi/delta v_j
        """
        mStar = 0
        deltaPhi = self.lmbda * V[j, :]
        deltaAlpha = numpy.zeros(self.k)
         
        for i in range(X.shape[0]): 
            omegai = omegaList[i]
            
            deltaBeta = numpy.zeros(self.k) 
            ui = U[i, :]
            
            if X[i, j] != 0: 
                omegaBari = numpy.setdiff1d(numpy.arange(X.shape[1]), omegai)
                
                p = j 
                vp = V[p, :]
                uivp = ui.dot(vp)
                kappa = exp(-uivp + self.r[i])
                onePlusKappa = 1+kappa
                
                for q in omegaBari: 
                    uivq = ui.dot(V[q, :])
                    gamma = exp(-uivp+uivq)
                    onePlusGamma = 1+gamma
                    
                    denom = onePlusGamma**2 * onePlusKappa**2
                    deltaBeta += ui*(gamma+kappa+2*gamma*kappa)/denom
            else:
                q = j 
                vq = V[q, :]
                uivq = ui.dot(vq)
                
                for p in omegai: 
                    uivp = ui.dot(V[p, :])
                    
                    gamma = exp(-uivp+uivq)
                    kappa = exp(-uivp+self.r[i])
                    
                    deltaBeta += -ui* gamma/((1+gamma)**2 * (1+kappa))
            
            numOmegai = omegai.shape[0]       
            numOmegaBari = X.shape[1]-numOmegai
            
            if numOmegai*numOmegaBari != 0: 
                deltaBeta /= float(numOmegai*numOmegaBari)
                mStar += 1
                
            deltaAlpha += deltaBeta 
        
        deltaAlpha /= float(mStar)
        deltaPhi -= deltaAlpha
        
        return deltaPhi
        
    def localAUC(self, X, U, V, omegaList): 
        """
        Compute the local AUC for the score functions UV^T relative to X with 
        quantile vector r. 
        """
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        localAuc = 0 
        mStar = 0
        allInds = numpy.arange(X.shape[1])
        
        for i in range(X.shape[0]): 
            omegai = omegaList[i]
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0                
                
                for p in omegai: 
                    for q in omegaBari: 
                        if Z[i, p] > Z[i, q] and Z[i, p] > self.r[i]: 
                            partialAuc += 1 
                            
                mStar += 1
                localAuc += partialAuc/float(omegai.shape[0] * omegaBari.shape[0])
        
        localAuc /= mStar        
        
        return localAuc

    def localAUCApprox(self, X, U, V, omegaList): 
        """
        Compute the estimated local AUC for the score functions UV^T relative to X with 
        quantile vector r. 
        """
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        localAuc = 0 
        mStar = 0
        allInds = numpy.arange(X.shape[1])
        sampleSize = self.numSamples

        for i in range(X.shape[0]): 
            omegai = omegaList[i]
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0                
                
                for j in range(sampleSize):
                    ind = numpy.random.randint(omegai.shape[0])
                    p = omegai[ind] 
                    
                    ind = numpy.random.randint(omegaBari.shape[0])
                    q = omegaBari[ind]   
                    
                    if Z[i, p] > Z[i, q] and Z[i, p] > self.r[i]: 
                        partialAuc += 1 
                            
                mStar += 1
                localAuc += partialAuc/float(sampleSize)
        
        localAuc /= mStar        
        
        return localAuc
    
    #@profile
    def objective(self, X, U, V, omegaList):         
        obj = 0 
        mStar = 0
        
        allInds = numpy.arange(X.shape[1])        
        
        for i in range(X.shape[0]): 
            omegai = omegaList[i]
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            ui = U[i, :]       
            uiV = ui.dot(V.T)
            ri = self.r[i]
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0                
                
                for p in omegai: 
                    uivp = uiV[p]
                    kappa = numpy.exp(-uivp+ri)
                    onePlusKappa = 1+kappa
                    
                    for q in omegaBari: 
                        uivq = uiV[q]
                        gamma = exp(-uivp+uivq)

                        partialAuc += 1/((1+gamma) * onePlusKappa)
                            
                mStar += 1
                obj += partialAuc/float(omegai.shape[0] * omegaBari.shape[0])
        
        obj /= mStar       
        obj = 0.5*self.lmbda * (numpy.sum(U**2) + numpy.sum(V**2)) - obj
        
        return obj 

    #@profile
    def objectiveApprox(self, X, U, V, omegaList):         
        obj = 0 
        mStar = 0
        
        allInds = numpy.arange(X.shape[1])        
        
        for i in range(X.shape[0]): 
            omegai = omegaList[i]
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            ui = U[i, :]       
            uiV = ui.dot(V.T)
            ri = self.r[i]
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0                
                
                indsP = numpy.random.randint(0, omegai.shape[0], self.numSamples)  
                indsQ = numpy.random.randint(0, omegaBari.shape[0], self.numSamples)
                
                for j in range(self.numSamples):
                    
                    p = omegai[indsP[j]] 
                    q = omegaBari[indsQ[j]]                  
                
                    uivp = uiV[p]
                    kappa = numpy.exp(-uivp+ri)
                    onePlusKappa = 1+kappa
                    
                    uivq = uiV[q]
                    gamma = exp(-uivp+uivq)

                    partialAuc += 1/((1+gamma) * onePlusKappa)
                            
                mStar += 1
                obj += partialAuc/float(self.numSamples)
        
        obj /= mStar       
        obj = 0.5*self.lmbda * (numpy.sum(U**2) + numpy.sum(V**2)) - obj
        
        return obj 