import os
import sys
from sandbox.recommendation.BprRecommender import BprRecommender 
from sandbox.util.SparseUtils import SparseUtils
import numpy
import unittest
import logging
import numpy.linalg 
import numpy.testing as nptst 
import sklearn.metrics 

class BprRecommenderTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(precision=3, suppress=True, linewidth=150)
        
        numpy.seterr(all="raise")
        numpy.random.seed(21)
    
    def testLearnModel(self): 
        m = 50 
        n = 20 
        k = 5
        u = 0.1 
        w = 1-u
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, w)
        
        lmbda = 0.1 
        gamma = 0.01
        learner = BprRecommender(k, lmbda, gamma)
        learner.max_iters = 50
        
        learner.learnModel(X)
        Z = learner.predict(n)
        
    def testModelSelect(self): 
        m = 50 
        n = 50 
        k = 5
        u = 0.5 
        w = 1-u
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, w)
        
        os.system('taskset -p 0xffffffff %d' % os.getpid())
        
        u = 0.2
        lmbda = 0.1 
        gamma = 0.01
        maxLocalAuc = BprRecommender(k, lmbda, gamma)
        maxLocalAuc.maxIterations = 20
        #maxLocalAuc.numProcesses = 1 
        
        maxLocalAuc.modelSelect(X)

    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()