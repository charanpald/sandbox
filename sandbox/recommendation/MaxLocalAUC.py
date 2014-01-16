
import numpy 
import sppy 


class MaxLocalAUC(object): 
    """
    Let's try different ways of maximising the local AUC with a penalty term. 
    """
    def __init__(self, lmbda, k, r, sigma=0.05, eps=0.1): 
        self.lmbda = k
        self.k = k 
        self.r = r
        self.sigma = sigma
        self.eps = eps 
    
    def learnModel(self, X): 
        """
        Max local AUC with Frobenius norm penalty. Solve with gradient descent. 
        The input is a sppy.csarray object 
        """
        
        m = X.shape[0]
        n = X.shape[1]
        
        rowInds, colInds = X.nonzero()
        mStar = numpy.unique(rowInds).shape[0]
        print(mStar)
        
        U = numpy.random.rand(m, self.k)
        V = numpy.random.rand(n, self.k)
        #U = numpy.zeros((m, self.k))
        #V = numpy.zeros((n, self.k))
        
        lastU = U+numpy.ones((m, self.k))*self.eps
        lastV = V+numpy.ones((n, self.k))*self.eps 
        
        normDeltaU = numpy.linalg.norm(U - lastU)
        normDeltaV = numpy.linalg.norm(V - lastV)
        
        while normDeltaU > self.eps and normDeltaV > self.eps: 
            lastU = U.copy() 
            lastV = V.copy() 
            
            deltaU = self.sigma*self.derivativeU(X, U, V, mStar)
            deltaV = self.sigma*self.derivativeV(X, U, V)
            
            U = U - deltaU
            V = V - deltaV
            
            print(self.localAUC(X, U, V), self.objective(X, U, V))
            
            normDeltaU = numpy.linalg.norm(U - lastU)
            normDeltaV = numpy.linalg.norm(V - lastV) 
            
            print(numpy.linalg.norm(U), numpy.linalg.norm(V), normDeltaU, normDeltaV)
            
            
        return U, V
        
        
    def derivativeU(self, X, U, V, mStar): 
        """
        Find the derivative for all of U. 
        """
        dU = numpy.zeros(U.shape)

        for i in range(U.shape[0]): 
            dU[i, :] = self.derivativeUi(X, U, V, i, mStar)
        
        return dU 
        
    def derivativeUi(self, X, U, V, i, mStar): 
        """
        delta phi/delta u_i
        """
        deltaPhi = self.lmbda * U[i, :]
        
        omegai = X[i, :].nonzero()[1]
        omegaBari = numpy.setdiff1d(numpy.arange(X.shape[1]), omegai)
        
        deltaAlpha = numpy.zeros(self.k)
        
        for p in omegai: 
            for q in omegaBari: 
                uivp = U[i, :].dot(V[p, :])
                uivq = U[i, :].dot(V[q, :])
                
                gamma = numpy.exp(-(uivp - uivq))
                kappa = numpy.exp(-(uivp - self.r[i]))
                
                deltaAlpha += (V[p, :] - V[q, :]) * gamma/((1+gamma)**2 * (1+kappa)) + V[q, :]*  kappa/((1+gamma) * (1+kappa)**2)
                
        if omegai.shape[0] * omegaBari.shape[0] != 0: 
            deltaAlpha /= float(omegai.shape[0] * omegaBari.shape[0]*mStar)
            
        deltaPhi -= deltaAlpha
        
        return deltaPhi
        
    def derivativeV(self, X, U, V): 
        """
        Find the derivative for all of V. 
        """
        dV = numpy.zeros(V.shape)

        for i in range(V.shape[0]): 
            dV[i, :] = self.derivativeVi(X, U, V, i)
        
        return dV         
        
    def derivativeVi(self, X, U, V, j): 
        """
        delta phi/delta v_j
        """
        mStar = 0
        deltaPhi = self.lmbda * V[j, :]
        deltaAlpha = numpy.zeros(self.k)
         
        
        for i in range(X.shape[0]): 
            omegai = X[i, :].nonzero()[1]
            omegaBari = numpy.setdiff1d(numpy.arange(X.shape[1]), omegai)
            
            deltaBeta = numpy.zeros(self.k)           
            
            if X[i, j] != 0: 
                p = j 
                
                for q in omegaBari: 
                    uivp = U[i, :].dot(V[p, :])
                    uivq = U[i, :].dot(V[q, :])
                    
                    gamma = numpy.exp(-(uivp - uivq))
                    kappa = numpy.exp(-(uivp - self.r[i]))
                    
                    deltaBeta += U[i, :] * (gamma/((1+gamma)**2 * (1+kappa)) + kappa/((1+gamma) * (1+kappa)**2))
            else:
                q = j 
                
                for p in omegai: 
                    uivp = U[i, :].dot(V[p, :])
                    uivq = U[i, :].dot(V[q, :])
                    
                    gamma = numpy.exp(-(uivp - uivq))
                    kappa = numpy.exp(-(uivp - self.r[i]))
                    deltaBeta += -U[i, :] * gamma/((1+gamma)**2 * (1+kappa))
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                deltaBeta /= float(omegai.shape[0] * omegaBari.shape[0])
                mStar += 1
                
            deltaAlpha += deltaBeta 
        
        deltaAlpha /= float(mStar)
        deltaPhi -= deltaAlpha
        
        return deltaPhi
        
    def localAUC(self, X, U, V): 
        """
        Compute the local AUC for the score functions UV^T relative to X with 
        quantile vector r. 
        """
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        localAuc = 0 
        mStar = 0
        
        for i in range(X.shape[0]): 
            omegai = X[i, :].nonzero()[1]
            omegaBari = numpy.setdiff1d(numpy.arange(X.shape[1]), omegai)
            
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
    
    def objective(self, X, U, V):         
        localAuc = 0 
        mStar = 0
        
        for i in range(X.shape[0]): 
            omegai = X[i, :].nonzero()[1]
            omegaBari = numpy.setdiff1d(numpy.arange(X.shape[1]), omegai)
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0                
                
                for p in omegai: 
                    for q in omegaBari: 
                        uivp = U[i, :].dot(V[p, :])
                        uivq = U[i, :].dot(V[q, :])
                        
                        gamma = numpy.exp(-(uivp - uivq))
                        kappa = numpy.exp(-(uivp - self.r[i]))
                        
                        partialAuc += 1/((1+gamma) * (1+kappa))
                            
                mStar += 1
                localAuc += partialAuc/float(omegai.shape[0] * omegaBari.shape[0])
        
        localAuc /= mStar        
        
        return localAuc 
            