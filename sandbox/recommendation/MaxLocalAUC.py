
import numpy 
import sppy 


class MaxLocalAUC(object): 
    """
    Let's try different ways of maximising the local AUC with a penalty term. 
    """
    def __init__(self, lmbda, k): 
        self.lmbda = k
        self.k = k 
    
    def learnModel(self, X): 
        """
        Max local AUC with Frobenius norm penalty. Solve with gradient descent. 
        The input is a sppy.csarray object 
        """
        
        m = X.shape[0]
        n = X.shape[1]
        
        U = numpy.random.rand(X.shape[0], self.k)
        V = numpy.random.rand(X.shape[1], self.k)
        
    def derivativeU(self, X, U, V, r, i): 
        """
        delta phi/delta u_i
        """
        
        deltaPhi = self.lmbda * U[i, :]
        
        omegai = X[i, :].nonzero()[0]
        omegaBari = numpy.setdiff1d(numpy.arange(X.shape[1]), omegai)
        
        deltaAlpha = numpy.zeros(self.k)
        
        for p in omegai: 
            for q in omegaBari: 
                uivp = U[i, :].dot(V[p, :])
                uivq = U[i, :].dot(V[q, :])
                
                e1 = numpy.exp(-(uivp - uivq))
                e2 = numpy.exp(-(uivp - r[i]))
                
                deltaAlpha += (V[p, :] - V[q, :]) * e1/((1+e1)**2 * (1+e2))
                
        deltaAlpha /= omegai.shape[0] * omegaBari.shape[0]
        deltaPhi -= deltaAlpha
        
        return deltaPhi
        
    def derivativeV(self, X, U, V, r, j): 
        """
        delta phi/delta v_j
        """
        deltaPhi = self.lmbda * V[i, :]
        
        for in range(X.shape[0]): 
            omegai = X[i, :].nonzero()[0]
            omegaBari = numpy.setdiff1d(numpy.arange(X.shape[1]), omegai)
            
            deltaAlpha = numpy.zeros(self.k)
            
            if X[i, j] != 0: 
                p = j 
                
                for q in omegaBari: 
                    uivp = U[i, :].dot(V[p, :])
                    uivq = U[i, :].dot(V[q, :])
                    
                    e1 = numpy.exp(-(uivp - uivq))
                    e2 = numpy.exp(-(uivp - r[i]))
                    
                    deltaAlpha += U[i, :] * e1/((1+e1)**2 * (1+e2))
        