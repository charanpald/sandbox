
import numpy 



class MaxLocalAUC(object): 
    """
    Let's try different ways of maximising the local AUC with a penalty term. 
    """
    def __init__(self, lmbda, k): 
        self.lmbda = k
        self.k = k 
    
    def frobeniusNorm(self, X): 
        """
        Max local AUC with Frobenius norm penalty. Solve with gradient descent. 
        """
        
        U = numpy.random.rand(X.shape[0], self.k)
        V = numpy.random.rand(X.shape[1], self.k)
        
        