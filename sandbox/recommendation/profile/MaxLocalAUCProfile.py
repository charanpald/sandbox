import numpy
import logging
import sys
from sandbox.util.ProfileUtils import ProfileUtils
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC, localAUCApprox
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.SparseUtilsCython import SparseUtilsCython

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class MaxLocalAUCProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        m = 500 
        n = 1000 
        self.k = 10 
        self.X = SparseUtils.generateSparseBinaryMatrix((m, n), self.k, csarray=True)
        
        
    def profileLearnModel(self):
        #Profile full gradient descent 
        m = 100 
        n = 50 
        self.k = 10 
        self.X = SparseUtils.generateSparseBinaryMatrix((m, n), self.k, csarray=True)    
    
        rho = 0.00001
        u = 0.5
        eps = 0.01
        sigma = 100
        maxLocalAuc = MaxLocalAUC(rho, self.k, u, sigma=sigma, eps=eps, stochastic=False)
        maxLocalAuc.maxIterations = 100
                
        ProfileUtils.profile('maxLocalAuc.learnModel(self.X)', globals(), locals())

    def profileLearnModel2(self):
        #Profile stochastic case 
        rho = 0.00
        u = 0.2
        w = 1-u
        eps = 10**-6
        alpha = 0.1
        k = self.k
        maxLocalAuc = MaxLocalAUC(k, w, alpha=alpha, eps=eps, stochastic=True)
        maxLocalAuc.numRowSamples = 100
        maxLocalAuc.numAucSamples = 10
        maxLocalAuc.maxIterations = 1000
        maxLocalAuc.numRecordAucSamples = 100
        maxLocalAuc.initialAlg = "rand"
        maxLocalAuc.rate = "optimal"
                
        trainX, testX = SparseUtils.splitNnz(self.X, 0.5)     

        def run(): 
            U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, time = maxLocalAuc.learnModel(trainX, True)  
            #logging.debug("Train Precision@5=" + str(MCEvaluator.precisionAtK(trainX, U, V, 5)))
            #logging.debug("Train Precision@10=" + str(MCEvaluator.precisionAtK(trainX, U, V, 10)))
            #logging.debug("Train Precision@20=" + str(MCEvaluator.precisionAtK(trainX, U, V, 20)))
            #logging.debug("Train Precision@50=" + str(MCEvaluator.precisionAtK(trainX, U, V, 50)))            
            
            #logging.debug("Test Precision@5=" + str(MCEvaluator.precisionAtK(testX, U, V, 5)))
            #logging.debug("Test Precision@10=" + str(MCEvaluator.precisionAtK(testX, U, V, 10)))
            #logging.debug("Test Precision@20=" + str(MCEvaluator.precisionAtK(testX, U, V, 20)))
            #logging.debug("Test Precision@50=" + str(MCEvaluator.precisionAtK(testX, U, V, 50)))
                
        ProfileUtils.profile('run()', globals(), locals())
     
    def profileLocalAucApprox(self): 
        m = 500 
        n = 1000 
        k = 10 
        X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m, n), k, csarray=True, verbose=True)
        
        u = 0.1
        w = 1-u
        numAucSamples = 200        
        
        omegaList = SparseUtils.getOmegaList(X)
        r = SparseUtilsCython.computeR(U, V, w, numAucSamples) 
        
        numRuns = 10        
        
        def run(): 
            for i in range(numRuns):
                localAUCApprox(X, U, V, omegaList, numAucSamples, r)
        
        ProfileUtils.profile('run()', globals(), locals())

profiler = MaxLocalAUCProfile()
#profiler.profileLearnModel()  
profiler.profileLearnModel2()
#profiler.profileLocalAucApprox()