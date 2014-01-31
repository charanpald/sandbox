
import sys
from sandbox.recommendation.MaxLocalAUCCython import localAUCApprox, updateUApprox, updateVApprox 
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
    
    @unittest.skip("")    
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
       
    #@unittest.skip("")
    def testUpdateUApprox(self): 
        """
        We'll test the case in which we approximate using a large number of samples 
        for the AUC and see if we get close to the exact derivative 
        """
        m = 20 
        n = 30 
        k = 2 
        numInds = 60
        X, U, s, V = SparseUtils.generateSparseLowRank((m, n), k, numInds, verbose=True)
        
        X = X/X

        r = numpy.ones(m)*0
        lmbda = 0.0
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(lmbda, k, r, eps=eps)
        maxLocalAuc.numAucSamples = m*n
        maxLocalAuc.sigma = 1
        maxLocalAuc.iterationsPerUpdate = m
        
        omegaList = maxLocalAuc.getOmegaList(X) 
        
        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)   

        lastU = U.copy()  
        
        #print(U)
        #print(V)
        rowInds = numpy.arange(m, dtype=numpy.uint)
        updateUApprox(X, U, V, omegaList, rowInds, maxLocalAuc.numAucSamples, maxLocalAuc.sigma, maxLocalAuc.lmbda, maxLocalAuc.r)

        dU = U-lastU        
        dU = numpy.diag(1/numpy.sqrt(numpy.diag(dU.dot(dU.T)))).dot(dU)
        #print(dU)
        
        #Let's compare against using the exact derivative 
        dU2 = numpy.zeros(U.shape)
        for i in range(X.shape[0]): 
            dU2[i, :] = -maxLocalAuc.derivativeUi(X, lastU, V, omegaList, i)
        
        dU2 = numpy.diag(1/numpy.sqrt(numpy.diag(dU2.dot(dU2.T)))).dot(dU2)
        #print(dU2)
        
        
        similarities = numpy.diag(dU.dot(dU2.T))
        nptst.assert_array_almost_equal(similarities, numpy.ones(m), 2)
        
        
    #@unittest.skip("")
    def testUpdateVApprox(self): 
        """
        We'll test the case in which we approximate using a large number of samples 
        for the AUC and see if we get close to the exact derivative 
        """
        m = 20 
        n = 30 
        k = 2 
        numInds = 60
        X, U, s, V = SparseUtils.generateSparseLowRank((m, n), k, numInds, verbose=True)
        
        X = X/X

        r = numpy.ones(m)*0
        lmbda = 0.0
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(lmbda, k, r, eps=eps)
        maxLocalAuc.numAucSamples = m*n
        maxLocalAuc.sigma = 1
        maxLocalAuc.iterationsPerUpdate = m
        
        omegaList = maxLocalAuc.getOmegaList(X) 
        
        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)   

         
        #Lets compute derivatives col by col
        for i in range(n): 
            lastV = V.copy() 
            
            colInds = numpy.array([i], numpy.uint)
            rowInds = numpy.arange(m, dtype=numpy.uint)
            updateVApprox(X, U, V, omegaList, rowInds, colInds, maxLocalAuc.numAucSamples, maxLocalAuc.sigma, maxLocalAuc.lmbda, maxLocalAuc.r) 

            dV = V[i, :] - lastV[i, :]   
            dV2 = -maxLocalAuc.derivativeVi(X, U, lastV, omegaList, i)

            dV = dV/numpy.linalg.norm(dV)
            dV2 = dV2/numpy.linalg.norm(dV2)

            similarity = dV.dot(dV2)
            
            self.assertAlmostEquals(similarity, 1, 3)
      
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()