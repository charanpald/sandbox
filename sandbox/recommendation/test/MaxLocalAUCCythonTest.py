
import sys
from sandbox.recommendation.MaxLocalAUCCython import derivativeUiApprox, derivativeUi, derivativeViApprox, derivativeVi
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



        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()