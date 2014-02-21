import numpy
import logging
import sys
from sandbox.util.ProfileUtils import ProfileUtils
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator

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
        lmbda = 0.00001
        u = 0.5
        eps = 0.5
        sigma = 100
        maxLocalAuc = MaxLocalAUC(lmbda, self.k, u, sigma=sigma, eps=eps)
                
        ProfileUtils.profile('maxLocalAuc.learnModel(self.X)', globals(), locals())

    def profileLearnModel2(self):
        #Profile stochastic case 
        rho = 0.00
        u = 0.2
        eps = 0.001
        sigma = 0.1
        k = self.k*10
        maxLocalAuc = MaxLocalAUC(rho, k, u, sigma=sigma, eps=eps, stochastic=True)
        maxLocalAuc.numRowSamples = 50
        maxLocalAuc.numColSamples = 50
        maxLocalAuc.numAucSamples = 100
        maxLocalAuc.maxIterations = 100
        maxLocalAuc.initialAlg = "rand"
        maxLocalAuc.rate = "optimal"
                
        trainX, testX = SparseUtils.splitNnz(self.X, 0.5)     

        def run(): 
            U, V, objs, aucs, iterations, times = maxLocalAuc.learnModel(trainX, True)  
            logging.debug("Train Precision@5=" + str(MCEvaluator.precisionAtK(trainX, U, V, 5)))
            logging.debug("Train Precision@10=" + str(MCEvaluator.precisionAtK(trainX, U, V, 10)))
            logging.debug("Train Precision@20=" + str(MCEvaluator.precisionAtK(trainX, U, V, 20)))
            logging.debug("Train Precision@50=" + str(MCEvaluator.precisionAtK(trainX, U, V, 50)))            
            
            logging.debug("Test Precision@5=" + str(MCEvaluator.precisionAtK(testX, U, V, 5)))
            logging.debug("Test Precision@10=" + str(MCEvaluator.precisionAtK(testX, U, V, 10)))
            logging.debug("Test Precision@20=" + str(MCEvaluator.precisionAtK(testX, U, V, 20)))
            logging.debug("Test Precision@50=" + str(MCEvaluator.precisionAtK(testX, U, V, 50)))
                
        ProfileUtils.profile('run()', globals(), locals())
        

profiler = MaxLocalAUCProfile()
#profiler.profileLearnModel()  
profiler.profileLearnModel2()