
import sys
from sandbox.recommendation.MaxLocalAUCCython import localAUCApprox, derivativeUiApprox, derivativeUi, derivativeViApprox, derivativeVi
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC 
from sandbox.util.SparseUtils import SparseUtils
import numpy
import unittest
import logging
import numpy.linalg 
import numpy.testing as nptst 

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
    def testDerivativeUiApprox(self): 
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
        rho = 0.0
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(rho, k, r, eps=eps)
        maxLocalAuc.numAucSamples = m*n
        maxLocalAuc.sigma = 1
        maxLocalAuc.iterationsPerUpdate = m
        
        omegaList = SparseUtils.getOmegaList(X) 
        
        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)   

        #Let's compare against using the exact derivative 
        for i in range(X.shape[0]): 
            du1 = derivativeUiApprox(X, U, V, omegaList, i, maxLocalAuc.numAucSamples, maxLocalAuc.getLambda(X), r, 1)
            du2 = derivativeUi(X, U, V, omegaList, i, maxLocalAuc.getLambda(X), r)
            self.assertTrue(numpy.linalg.norm(du1 - du2) < 0.1)
            
        maxLocalAuc.rho = 0.1
        errors = numpy.zeros(m)     
        
        for i in range(X.shape[0]): 
            du1 = derivativeUiApprox(X, U, V, omegaList, i, maxLocalAuc.numAucSamples, maxLocalAuc.getLambda(X), r, 1)
            du2 = derivativeUi(X, U, V, omegaList, i, maxLocalAuc.getLambda(X), r)
            errors[i] = numpy.linalg.norm(du1 - du2)
            #self.assertTrue(numpy.linalg.norm(du1 - du2) < 0.1)

        self.assertTrue((errors < 0.1).sum() < n*3/4.0)        
        
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
        rho = 0.0
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(rho, k, r, eps=eps)
        maxLocalAuc.numAucSamples = 30
        maxLocalAuc.numRowSamples = m
        maxLocalAuc.sigma = 1

        
        omegaList = SparseUtils.getOmegaList(X) 
        
        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)   

        errors = numpy.zeros(n)        
        
        #Lets compute derivatives col by col
        #Note in some cases we get a huge error but in general it is low 
        #So check if most errors are small 
        for i in range(n): 
            dv1 = derivativeViApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.getLambda(X), r, 1)
            dv2 = derivativeVi(X, U, V, omegaList, i, maxLocalAuc.getLambda(X), r)
            errors[i] += numpy.linalg.norm(dv1 - dv2)
      
        self.assertTrue((errors < 0.1).sum() < n*3/4.0)
        
        maxLocalAuc.rho = 0.1
        errors = numpy.zeros(n)
        
        for i in range(n): 
            dv1 = derivativeViApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.getLambda(X), r, 1)
            dv2 = derivativeVi(X, U, V, omegaList, i, maxLocalAuc.getLambda(X), r)
            errors[i] += numpy.linalg.norm(dv1 - dv2)
      
        self.assertTrue((errors < 0.1).sum() < n*3/4.0)
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()