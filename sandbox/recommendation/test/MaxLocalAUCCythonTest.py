
import sys
from sandbox.recommendation.MaxLocalAUCCython import derivativeUiApprox, derivativeUi, derivativeViApprox, derivativeVi, objectiveApprox, objective
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
        We'll test the case in which we apprormate using a large number of samples 
        for the AUC and see if we get close to the exact derivative 
        """
        m = 20 
        n = 30 
        k = 3 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, csarray=True)
        
        w = 0.1
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(k, w, eps=eps)
        maxLocalAuc.numAucSamples = n**2
        maxLocalAuc.numRowSamples = m 
        maxLocalAuc.lmbda = 0
        maxLocalAuc.rho = 0 

        
        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)
        
        numRuns = 500 
        numTests = 5
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
        r = numpy.zeros(m)
        colIndsProbs = numpy.ones(colInds.shape[0])
        
        for i in range(m): 
            colIndsProbs[indPtr[i]:indPtr[i+1]] /= colIndsProbs[indPtr[i]:indPtr[i+1]].sum()
            colIndsProbs[indPtr[i]:indPtr[i+1]] = numpy.cumsum(colIndsProbs[indPtr[i]:indPtr[i+1]])

        #Let's compare against using the exact derivative 
        for i in numpy.random.permutation(m)[0:numTests]:  
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(indPtr, colInds, colIndsProbs, U, V, r, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(indPtr, colInds, U, V, r, i, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            #print(du1/numpy.linalg.norm(du1), du2/numpy.linalg.norm(du2))
            nptst.assert_array_almost_equal(du1, du2, 3)
            
        maxLocalAuc.rho = 0.1

        for i in numpy.random.permutation(m)[0:numTests]:  
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(indPtr, colInds, colIndsProbs, U, V, r, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(indPtr, colInds, U, V, r, i, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(du1, du2, 3)
            
        maxLocalAuc.lmbda = 0.5 
        
        for i in numpy.random.permutation(m)[0:numTests]:  
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(indPtr, colInds, colIndsProbs, U, V, r, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(indPtr, colInds, U, V, r, i, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(du1, du2, 3)
            
        maxLocalAuc.numAucSamples = 10
        numRuns = 1000
        
        for i in numpy.random.permutation(m)[0:numTests]:  
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(indPtr, colInds, colIndsProbs, U, V, r, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(indPtr, colInds, U, V, r, i, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(du1, du2, 3)
        
    @unittest.skip("")
    def testDerivativeViApprox(self): 
        """
        We'll test the case in which we apprormate using a large number of samples 
        for the AUC and see if we get close to the exact derivative 
        """
        m = 20 
        n = 30 
        k = 3 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, csarray=True)
        
        w = 0.1
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(k, w, eps=eps)
        maxLocalAuc.numAucSamples = n
        maxLocalAuc.numRowSamples = m 
        maxLocalAuc.lmbda = 0
        maxLocalAuc.rho = 0 
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
        r = numpy.random.rand(m)
        colIndsProbs = numpy.ones(colInds.shape[0])
        
        for i in range(m): 
            colIndsProbs[indPtr[i]:indPtr[i+1]] /= colIndsProbs[indPtr[i]:indPtr[i+1]].sum()
            colIndsProbs[indPtr[i]:indPtr[i+1]] = numpy.cumsum(colIndsProbs[indPtr[i]:indPtr[i+1]])
        
        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)
        
        numRuns = 100 
        numTests = 5

        #Let's compare against using the exact derivative 
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(indPtr, colInds, colIndsProbs, U, V, r, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples,  maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(indPtr, colInds, U, V, r, i, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(dv1, dv2, 3)
            
        maxLocalAuc.rho = 0.2

        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(indPtr, colInds, colIndsProbs, U, V, r, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples,  maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(indPtr, colInds, U, V, r, i, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(dv1, dv2, 3)
            
        maxLocalAuc.lmbda = 0.5 
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
    
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(indPtr, colInds, colIndsProbs, U, V, r, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples,  maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(indPtr, colInds, U, V, r, i, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(dv1, dv2, 3)
            
        maxLocalAuc.numRowSamples = 10 
        numRuns = 5000
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(indPtr, colInds, colIndsProbs, U, V, r, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples,  maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(indPtr, colInds, U, V, r, i, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(dv1, dv2, 3)

        maxLocalAuc.numRowSamples = m 
        maxLocalAuc.numAucSamples = 10 
        numRuns = 5000
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(indPtr, colInds, colIndsProbs, U, V, r, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples,  maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(indPtr, colInds, U, V, r, i, maxLocalAuc.lmbda, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(dv1, dv2, 3)


    #@unittest.skip("")
    def testObjectiveApprox(self): 
        """
        We'll test the case in which we apprormate using a large number of samples 
        for the AUC and see if we get close to the exact objective 
        """
        m = 20 
        n = 30 
        k = 3 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, csarray=True)
        
        w = 0.1
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(k, w, eps=eps)
        maxLocalAuc.numAucSamples = n
        maxLocalAuc.numRowSamples = m 
        maxLocalAuc.lmbda = 0
        maxLocalAuc.rho = 0 
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
        r = numpy.random.rand(m)
        colIndsProbs = numpy.ones(colInds.shape[0])
        
        for i in range(m): 
            colIndsProbs[indPtr[i]:indPtr[i+1]] /= colIndsProbs[indPtr[i]:indPtr[i+1]].sum()
            colIndsProbs[indPtr[i]:indPtr[i+1]] = numpy.cumsum(colIndsProbs[indPtr[i]:indPtr[i+1]])
        
        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)
        
        numRuns = 500 
        numTests = 5

        #Let's compare against using the exact derivative 
        for i in range(numTests): 
            obj = 0
            for j in range(numRuns): 
                obj += objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.numAucSamples, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            obj /= numRuns
            
            obj2 = objective(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)    
            self.assertAlmostEquals(obj, obj2, 2)
            
        maxLocalAuc.rho = 0.2

        for i in range(numTests): 
            obj = 0
            for j in range(numRuns): 
                obj += objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.numAucSamples, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            obj /= numRuns
            
            obj2 = objective(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)    
            self.assertAlmostEquals(obj, obj2, 2)

        maxLocalAuc.lmbda = 0.2

        for i in range(numTests): 
            obj = 0
            for j in range(numRuns): 
                obj += objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.numAucSamples, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
            obj /= numRuns
            
            obj2 = objective(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)    
            self.assertAlmostEquals(obj, obj2, 2)
        
        #Check full and summary versions are the same 
        obj = objective(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.lmbda, maxLocalAuc.rho, True) 
        obj = obj.mean()
        obj2 = objective(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False) 
        self.assertAlmostEquals(obj, obj2, 2)
        
        
        U = numpy.zeros((X.shape[0], k))
        V = numpy.zeros((X.shape[1], k))
        r = numpy.zeros(X.shape[0])
        
        maxLocalAuc.rho = 0 
        obj = 0
        for j in range(numRuns): 
            obj += objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.numAucSamples, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
        obj /= numRuns
        
        obj2 = objective(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
        
        self.assertEquals(obj, 0.5)
        self.assertEquals(obj2, 0.5)
        
        maxLocalAuc.rho = 1
        r = numpy.ones(X.shape[0])
        obj = 0
        for j in range(numRuns): 
            obj += objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, r, maxLocalAuc.numAucSamples, maxLocalAuc.lmbda, maxLocalAuc.rho, False)
        obj /= numRuns        
        
        self.assertAlmostEquals(obj, 2.0)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()