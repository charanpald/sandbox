
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC 
from sandbox.util.SparseUtils import SparseUtils
import numpy
import sppy 
import unittest
import logging
import scipy.sparse 
import numpy.linalg 

class MaxLocalAUCTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(precision=3, suppress=True, linewidth=100)
        
        numpy.seterr(all="raise")
        numpy.random.seed(21)
    
    def testLearnModel(self): 
        m = 50 
        n = 20 
        k = 5 
        numInds = 500
        X = SparseUtils.generateSparseLowRank((m, n), k, numInds)
        
        X = X/X
        
        lmbda = 0.00
        r = numpy.ones(m)*0.0
        maxLocalAuc = MaxLocalAUC(lmbda, k, r)
        
        U, V = maxLocalAuc.learnModel(X)
        
        print(U)
        print(V)
                
        
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()