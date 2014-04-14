"""
Wrapper to use gamboviol implementation of CLiMF.
"""

import numpy 
import sppy
import logging
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling

try:
    from climf import update
except:
    logging.warning("climf not installed, cannot be used later on")
    def update(X,U,V,lmbda,gamma):
        raise NameError("'climf' is not installed, so 'update' is not defined")
        
        

class Climf(object): 
    """
    An interface to use CLiMF recommender system. 
    """
    
    def __init__(self, k, lmbda, gamma): 
        self.k = k 
        self.lmbda = lmbda
        self.gamma = gamma
                
        self.max_iters = 25
        
    def learnModel(self, X): 
        self.X=X
        self.U = 0.01*numpy.random.random_sample((X.shape[0],self.k))
        self.V = 0.01*numpy.random.random_sample((X.shape[1],self.k))

        for it in xrange(self.max_iters):
            logging.debug("Iteration " + str(it))
            update(self.X,self.U,self.V,self.lmbda,self.gamma)
    
    def predict(self, maxItems): 
        return MCEvaluator.recommendAtk(self.U, self.V, maxItems)
    
    def modelSelect(self, X): 
        """
        Perform model selection on X and return the best parameters. 
        """
        cat("modelSelect is not implemented for CLiMF")
        stop()
            
    def copy(self): 
        learner = CLiMF(self.k, self.lmbda, self.gamma)
        learner.max_iters = self.max_iters
        
        return learner 

    def __str__(self): 
        outputStr = "CLiMF Recommender: k=" + str(self.k) 
        outputStr += " lambda=" + str(self.lmbda)
        outputStr += " gamma=" + str(self.gamma)
        outputStr += " max iters=" + str(self.max_iters)
        
        return outputStr         

def main():
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    matrixFileName = PathDefaults.getDataDir() + "movielens/ml-100k/u.data" 
    data = numpy.loadtxt(matrixFileName)
    X = sppy.csarray((numpy.max(data[:, 0]), numpy.max(data[:, 1])), storagetype="row")
    X[data[:, 0]-1, data[:, 1]-1] = numpy.array(data[:, 2]>3, numpy.int)
    logging.debug("Read file: " + matrixFileName)
    logging.debug("Shape of data: " + str(X.shape))
    logging.debug("Number of non zeros " + str(X.nnz))
    
    u = 0.1 
    w = 1-u
    (m, n) = X.shape

    testSize = 5
    trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
    trainX, testX = trainTestXs[0]
    trainX = trainX.toScipyCsr()

    learner = Climf(k=20, lmbda=0.001, gamma=0.0001)
    learner.learnModel(trainX)
    
if __name__=='__main__':
    main()


