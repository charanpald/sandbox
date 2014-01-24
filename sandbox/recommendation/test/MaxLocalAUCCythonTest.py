
import sys
from sandbox.recommendation.MaxLocalAUCCython import localAUCApprox 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC 
from sandbox.util.SparseUtils import SparseUtils
import numpy
import unittest
import logging
import numpy.linalg 
import numpy.testing as nptst 
import sklearn.metrics 

class MaxLocalAUCTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(precision=3, suppress=True, linewidth=150)
        
        numpy.seterr(all="raise")
        numpy.random.seed(21)
        
    def testLocalAucApprox(self): 
        m = 100 
        n = 200 
        k = 2 
        numInds = 100
        X, U, s, V = SparseUtils.generateSparseLowRank((m, n), k, numInds, verbose=True)
        
        X = X/X

        r = numpy.ones(m)*-10
        lmbda = 0.0
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(lmbda, k, r, eps=eps)
        
        omegaList = maxLocalAuc.getOmegaList(X)
        
        localAuc = maxLocalAuc.localAUC(X, U, V, omegaList)
        
        samples = numpy.arange(50, 200, 10)
        
        for i, sampleSize in enumerate(samples): 
            maxLocalAuc.numAucSamples = sampleSize
            localAuc2 = localAUCApprox(X, U, V, omegaList, sampleSize, r)

            self.assertAlmostEqual(localAuc2, localAuc, 1)
        
        #Test more accurately 
        sampleSize = 1000
        localAuc2 = localAUCApprox(X, U, V, omegaList, sampleSize, r)
        self.assertAlmostEqual(localAuc2, localAuc, 2)
        
        #Now set a high r 
        Z = U.dot(V.T)
        r = numpy.ones(m)*0
        maxLocalAuc.r = r
        localAuc = maxLocalAuc.localAUC(X, U, V, omegaList)   

        for i, sampleSize in enumerate(samples): 
            maxLocalAuc.numAucSamples = sampleSize
            localAuc2 = localAUCApprox(X, U, V, omegaList, sampleSize, r)

            self.assertAlmostEqual(localAuc2, localAuc, 1)
            
        #Test more accurately 
        sampleSize = 1000
        localAuc2 = localAUCApprox(X, U, V, omegaList, sampleSize, r)
        self.assertAlmostEqual(localAuc2, localAuc, 2)
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()