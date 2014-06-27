import os
import sys
from sandbox.recommendation.CLiMF import CLiMF 
from sandbox.util.SparseUtils import SparseUtils
import numpy
import unittest
import logging
import numpy.linalg 
import numpy.testing as nptst 
import sklearn.metrics 

class CLiMFTest(unittest.TestCase):
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
        learner = CLiMF(k, lmbda, gamma)
        learner.max_iters = 50
        
        learner.learnModel(X)
        Z = learner.predict(n)
        
        #Bit weird that all rows are the same 
        print(Z)
        
    def testModelSelect(self): 
        m = 50 
        n = 20 
        k = 5
        u = 0.1 
        w = 1-u
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, w)
        
        lmbda = 0.1 
        gamma = 0.01
        learner = CLiMF(k, lmbda, gamma)
        learner.max_iters = 10
        
        learner.modelSelect(X)

    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()