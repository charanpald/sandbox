
import sys
from sandbox.recommendation.MaxLocalAUCCython import localAUCApprox, derivativeUiApprox, derivativeUi, derivativeViApprox, derivativeVi, inverseChoicePy, choicePy
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython 
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
       
    @unittest.skip("")
    def testDerivativeUiApprox(self): 
        """
        We'll test the case in which we approximate using a large number of samples 
        for the AUC and see if we get close to the exact derivative 
        """
        m = 20 
        n = 30 
        k = 3 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k)
        
        numAucSamplesR = 100 
        w = 0.1
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(k, w, eps=eps)
        maxLocalAuc.numAucSamples = n
        maxLocalAuc.numRowSamples = m 
        maxLocalAuc.lmbda = 0
        maxLocalAuc.rho = 0 
        maxLocalAuc.iterationsPerUpdate = m
        
        omegaList = SparseUtils.getOmegaList(X) 
        
        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)
        r =  SparseUtilsCython.computeR(U, V, 1-w, numAucSamplesR)
        
        numRuns = 500 
        numTests = 5

        #Let's compare against using the exact derivative 
        for i in numpy.random.permutation(m)[0:numTests]: 
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(X, U, V, omegaList, i, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(du1, du2, 3)
            
        maxLocalAuc.rho = 0.1

        for i in numpy.random.permutation(m)[0:numTests]: 
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(X, U, V, omegaList, i, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(du1, du2, 3)
            
        maxLocalAuc.lmbda = 0.5 
        
        for i in numpy.random.permutation(m)[0:numTests]:  
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(X, U, V, omegaList, i, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(du1, du2, 3)
            
        maxLocalAuc.numAucSamples = 10
        numRuns = 1000
        
        for i in numpy.random.permutation(m)[0:numTests]:  
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(X, U, V, omegaList, i, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(du1, du2, 3)
        
    @unittest.skip("")
    def testDerivativeViApprox(self): 
        """
        We'll test the case in which we approximate using a large number of samples 
        for the AUC and see if we get close to the exact derivative 
        """
        m = 20 
        n = 30 
        k = 3 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k)
        
        numAucSamplesR = 100 
        w = 0.1
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(k, w, eps=eps)
        maxLocalAuc.numAucSamples = n
        maxLocalAuc.numRowSamples = m 
        maxLocalAuc.lmbda = 0
        maxLocalAuc.rho = 0 
        maxLocalAuc.iterationsPerUpdate = m
        
        omegaList = SparseUtils.getOmegaList(X) 
        
        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)
        r =  SparseUtilsCython.computeR(U, V, 1-w, numAucSamplesR)
        
        numRuns = 100 
        numTests = 5

        #Let's compare against using the exact derivative 
        for i in numpy.random.permutation(m)[0:numTests]: 
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(X, U, V, omegaList, i, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            
            nptst.assert_array_almost_equal(dv1, dv2, 3)
            
        maxLocalAuc.rho = 0.1

        for i in numpy.random.permutation(m)[0:numTests]:  
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(X, U, V, omegaList, i, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(dv1, dv2, 3)
            
        maxLocalAuc.lmbda = 0.5 
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(X, U, V, omegaList, i, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(dv1, dv2, 3)
            

        maxLocalAuc.numRowSamples = 10 
        numRuns = 5000
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(X, U, V, omegaList, i, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)  
            nptst.assert_array_almost_equal(dv1, dv2, 3)

        maxLocalAuc.numRowSamples = m 
        maxLocalAuc.numAucSamples = 10 
        numRuns = 5000
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(X, U, V, omegaList, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(X, U, V, omegaList, i, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)  
            nptst.assert_array_almost_equal(dv1, dv2, 3)


    def testInverseChoicePy(self):
        n = 100
        a = numpy.array(numpy.random.randint(0, n, 50), numpy.int32)
        a = numpy.unique(a)

        numRuns = 100 
        for i in range(numRuns): 
            j = inverseChoicePy(a, n)
            self.assertTrue(j not in a)
        

    def testChoicePy(self): 
        n = 100
        k = 50
        a = numpy.array(numpy.random.randint(0, n, k), numpy.int32)
        a = numpy.unique(a)
        probs = numpy.ones(a.shape[0])/float(a.shape[0])
        
        sample = choicePy(a, 10, probs)

        for item in sample:
            self.assertTrue(item in a)

        probs = numpy.zeros(a.shape[0])
        probs[2] = 1
        sample = choicePy(a, 10, probs)
        
        for item in sample:
            self.assertEquals(item, a[2])
            
        a = numpy.array([0, 1, 2], numpy.int32)
        probs = numpy.array([0.2, 0.6, 0.2])
        
        runs = 10000
        sample = choicePy(a, runs, probs)
        
        nptst.assert_array_almost_equal(numpy.bincount(sample)/float(runs), probs, 2)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()