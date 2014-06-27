
import os
import sys
from sandbox.recommendation.WeightedMf import WeightedMf 
from sandbox.util.SparseUtils import SparseUtils
import numpy
import unittest
import logging
import numpy.linalg 
import numpy.testing as nptst 
import sklearn.metrics 

class WeightedMfTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(precision=3, suppress=True, linewidth=150)
        
        numpy.seterr(all="raise")
        numpy.random.seed(21)

    def testModelSelect(self): 
        m = 50 
        n = 50 
        k = 5
        u = 0.5 
        w = 1-u
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, w)
        
        os.system('taskset -p 0xffffffff %d' % os.getpid())
        

        learner = WeightedMf(k)
        learner.maxIterations = 10        
        learner.ks = 2**numpy.arange(3, 5)
        learner.folds = 2
        #maxLocalAuc.numProcesses = 1 
        
        learner.modelSelect(X)

    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()