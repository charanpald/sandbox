
import sys
from sandbox.recommendation.MaxAUCLogistic import MaxAUCLogistic
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython 
import numpy
import unittest
import logging
import numpy.linalg 
import numpy.testing as nptst 

class MaxAUCLogisticTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
        
        #numpy.seterr(all="raise")
        numpy.random.seed(21)


    @unittest.skip("")
    def testDerivativeU(self): 
        m = 10 
        n = 20 
        nnzPerRow = 5 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), nnzPerRow, csarray=True)
        
        k = 5
        eps = 0.05
        learner = MaxAUCLogistic(k)
        learner.normalise = False
        learner.lmbdaU = 0
        learner.lmbdaV = 0
        learner.rho = 1.0
        learner.numAucSamples = n

        numRuns = 20
        gi = numpy.random.rand(m)
        gi /= gi.sum()        
        gp = numpy.random.rand(n)
        gp /= gp.sum()        
        gq = numpy.random.rand(n)
        gq /= gq.sum()     
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)

        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)
            deltaU = numpy.zeros(U.shape)
            for i in range(X.shape[0]): 
                deltaU[i, :] = learner.derivativeUi(indPtr, colInds, U, V, gp, gq, i)      
    
            deltaU2 = numpy.zeros(U.shape) 
            eps = 10**-8         
            
            for i in range(m): 
                for j in range(k):
                    tempU = U.copy() 
                    tempU[i,j] += eps
                    obj1 = learner.objective(indPtr, colInds, indPtr, colInds, tempU, V, gp, gq)
                    
                    tempU = U.copy() 
                    tempU[i,j] -= eps
                    obj2 = learner.objective(indPtr, colInds, indPtr, colInds, tempU, V, gp, gq)
                    
                    deltaU2[i,j] = (obj1-obj2)/(2*eps)
    
                #deltaU2[i,:] = deltaU2[i,:]/numpy.linalg.norm(deltaU2[i,:])
            
            #print(deltaU*100)
            #print(deltaU2*100)
            nptst.assert_almost_equal(deltaU, deltaU2, 3)
        
        #Try r != 0 and rho > 0
        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)
            learner.rho = 0.1
            
            deltaU = numpy.zeros(U.shape)
            for i in range(X.shape[0]): 
                deltaU[i, :] = learner.derivativeUi(indPtr, colInds, U, V, gp, gq, i)
            
            deltaU2 = numpy.zeros(U.shape) 
            eps = 10**-9        
            
            for i in range(m): 
                for j in range(k):
                    tempU = U.copy() 
                    tempU[i,j] += eps
                    obj1 = learner.objective(indPtr, colInds, indPtr, colInds, tempU, V, gp, gq)
                    
                    tempU = U.copy() 
                    tempU[i,j] -= eps
                    obj2 = learner.objective(indPtr, colInds, indPtr, colInds, tempU, V, gp, gq)
                    
                    deltaU2[i,j] = (obj1-obj2)/(2*eps)
                                
            nptst.assert_almost_equal(deltaU, deltaU2, 3)
        
        #Try lmbda > 0
        
        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)
            learner.lmbdaU = 0.5
            
            deltaU = numpy.zeros(U.shape)
            for i in range(X.shape[0]): 
                deltaU[i, :] = learner.derivativeUi(indPtr, colInds, U, V, gp, gq, i) 
            
            deltaU2 = numpy.zeros(U.shape) 
            eps = 10**-9        
            
            for i in range(m): 
                for j in range(k):
                    tempU = U.copy() 
                    tempU[i,j] += eps
                    obj1 = learner.objective(indPtr, colInds, indPtr, colInds, tempU, V, gp, gq)
                    
                    tempU = U.copy() 
                    tempU[i,j] -= eps
                    obj2 = learner.objective(indPtr, colInds, indPtr, colInds, tempU, V, gp, gq)
                    
                    deltaU2[i,j] = (obj1-obj2)/(2*eps)
                                
            nptst.assert_almost_equal(deltaU, deltaU2, 3)
        
       
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
        learner = MaxAUCLogistic(k, w)
        learner.normalise = False
        learner.lmbdaU = 0
        learner.lmbdaV = 0
        learner.rho = 1.0
        learner.numAucSamples = 100

        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)

        gp = numpy.random.rand(n)
        gp /= gp.sum()        
        gq = numpy.random.rand(n)
        gq /= gq.sum()     

        
        numRuns = 200 
        numTests = 5
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
        permutedColInds = numpy.arange(n, dtype=numpy.uint32)

        #Test with small number of AUC samples, but normalise 
        learner.numAucSamples = n
        numRuns = 1000
        
        for i in numpy.random.permutation(m)[0:numTests]:  
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += learner.derivativeUiApprox(indPtr, colInds, U, V, gp, gq, permutedColInds, i)
            du1 /= numRuns
            du2 = learner.derivativeUi(indPtr, colInds, U, V, gp, gq, i) 
            #print(du1, du2)
            print(du1/numpy.linalg.norm(du1), du2/numpy.linalg.norm(du2))
            #print(numpy.linalg.norm(du1 - du2)/numpy.linalg.norm(du1))
            self.assertTrue(numpy.linalg.norm(du1 - du2)/numpy.linalg.norm(du1) < 0.5)

        #Let's compare against using the exact derivative 
        for i in numpy.random.permutation(m)[0:numTests]:  
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += learner.derivativeUiApprox(indPtr, colInds, U, V, gp, gq, permutedColInds, i)
            du1 /= numRuns
            du2 = learner.derivativeUi(indPtr, colInds, U, V, gp, gq, i)   
            
            print(du1/numpy.linalg.norm(du1), du2/numpy.linalg.norm(du2))
            nptst.assert_array_almost_equal(du1, du2, 2)
            
            
        learner.lmbdaV = 0.5 
        
        for i in numpy.random.permutation(m)[0:numTests]:  
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += learner.derivativeUiApprox(indPtr, colInds, U, V, gp, gq, permutedColInds, i)
            du1 /= numRuns
            du2 = learner.derivativeUi(indPtr, colInds, U, V, gp, gq, i)   
            nptst.assert_array_almost_equal(du1, du2, 2)
            print(du1/numpy.linalg.norm(du1), du2/numpy.linalg.norm(du2))
            
            
    @unittest.skip("")
    def testDerivativeV(self): 
        m = 10 
        n = 20 
        nnzPerRow = 5 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), nnzPerRow, csarray=True)
        
        for i in range(m):
            X[i, 0] = 1
            X[i, 1] = 0
        
        k = 5
        u = 0.1
        w = 1-u
        eps = 0.05
        learner = MaxAUCLogistic(k, w)
        learner.normalise = False
        learner.lmbdaU = 0
        learner.lmbdaV = 0
        learner.rho = 1.0
        learner.numAucSamples = 100

        numRuns = 20
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
            
        gp = numpy.random.rand(n)
        gp /= gp.sum()        
        gq = numpy.random.rand(n)
        gq /= gq.sum()            
        
        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)            
            
            deltaV = numpy.zeros(V.shape)
            for j in range(n): 
                deltaV[j, :] = learner.derivativeVi(indPtr, colInds, U, V, gp, gq, j)   
            
            deltaV2 = numpy.zeros(V.shape)    
            
            eps = 0.00001        
            
            for i in range(n): 
                for j in range(k):
                    tempV = V.copy() 
                    tempV[i,j] += eps
                    obj1 = learner.objective(indPtr, colInds, indPtr, colInds, U, tempV, gp, gq)
                    
                    tempV = V.copy() 
                    tempV[i,j] -= eps
                    obj2 = learner.objective(indPtr, colInds, indPtr, colInds, U, tempV, gp, gq)
                    
                    deltaV2[i,j] = (obj1-obj2)/(2*eps)
                #deltaV2[i,:] = deltaV2[i,:]/numpy.linalg.norm(deltaV2[i,:])                   
                        

            nptst.assert_almost_equal(deltaV, deltaV2, 3)

        #Try r != 0 and rho > 0
        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)   
            learner.rho = 1.0    
            
            deltaV = numpy.zeros(V.shape)
            for j in range(n): 
                deltaV[j, :] = learner.derivativeVi(indPtr, colInds, U, V, gp, gq, j)    
            
            deltaV2 = numpy.zeros(V.shape)
            
            for i in range(n): 
                for j in range(k):
                    tempV = V.copy() 
                    tempV[i,j] += eps
                    obj1 = learner.objective(indPtr, colInds, indPtr, colInds, U, tempV, gp, gq)
                    
                    tempV = V.copy() 
                    tempV[i,j] -= eps
                    obj2 = learner.objective(indPtr, colInds, indPtr, colInds, U, tempV, gp, gq)
                    
                    deltaV2[i,j] = (obj1-obj2)/(2*eps)
                #deltaV2[i,:] = deltaV2[i,:]/numpy.linalg.norm(deltaV2[i,:])
                           
            nptst.assert_almost_equal(deltaV, deltaV2, 3)
        
        
        #Try r != 0 and rho > 0
        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)              
            
            learner.lmbdaV = 100    
            
            deltaV = numpy.zeros(V.shape)
            for j in range(n): 
                deltaV[j, :] = learner.derivativeVi(indPtr, colInds, U, V, gp, gq, j)
            
            deltaV2 = numpy.zeros(V.shape)
            
            for i in range(n): 
                for j in range(k):
                    tempV = V.copy() 
                    tempV[i,j] += eps
                    obj1 = learner.objective(indPtr, colInds, indPtr, colInds, U, tempV, gp, gq)
                    
                    tempV = V.copy() 
                    tempV[i,j] -= eps
                    obj2 = learner.objective(indPtr, colInds, indPtr, colInds, U, tempV,  gp, gq)
                    
                    deltaV2[i,j] = (obj1-obj2)/(2*eps)
                #deltaV2[i,:] = deltaV2[i,:]/numpy.linalg.norm(deltaV2[i,:])
              
            nptst.assert_almost_equal(deltaV, deltaV2, 3)         
      
    #@unittest.skip("")
    def testDerivativeViApprox(self): 
        """
        We'll test the case in which we apprormate using a large number of samples 
        for the AUC and see if we get close to the exact derivative 
        """
        m = 20 
        n = 30 
        k = 3 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, csarray=True)
        
        for i in range(m):
            X[i, 0] = 1
            X[i, 1] = 0
        
        w = 0.1
        eps = 0.001
        learner = MaxAUCLogistic(k, w)
        learner.normalise = False
        learner.lmbdaU = 0
        learner.lmbdaV = 0
        learner.numAucSamples = n
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)

        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)
             
        gp = numpy.random.rand(n)
        gp /= gp.sum()        
        gq = numpy.random.rand(n)
        gq /= gq.sum()     
        
        permutedRowInds = numpy.array(numpy.random.permutation(m), numpy.uint32)
        permutedColInds = numpy.array(numpy.random.permutation(n), numpy.uint32)
        
        maxLocalAuc = MaxLocalAUC(k, w)
        normGp, normGq = maxLocalAuc.computeNormGpq(indPtr, colInds, gp, gq, m)
        
        numRuns = 200 
        numTests = 5

        #Let's compare against using the exact derivative 
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += learner.derivativeViApprox(indPtr, colInds, U, V, gp, gq, normGp, normGq, permutedRowInds, permutedColInds, i)
            dv1 /= numRuns
            dv2 = learner.derivativeVi(indPtr, colInds, U, V, gp, gq, i)   
            
            
            dv3 = numpy.zeros(k)
            for j in range(k): 
                eps = 10**-6
                tempV = V.copy() 
                tempV[i,j] += eps
                obj1 = learner.objective(indPtr, colInds, indPtr, colInds, U, tempV, gp, gq)
                
                tempV = V.copy() 
                tempV[i,j] -= eps
                obj2 = learner.objective(indPtr, colInds, indPtr, colInds, U, tempV, gp, gq)
                
                dv3[j] = (obj1-obj2)/(2*eps)            
            
            print(dv1, dv2, dv3)
            
            nptst.assert_array_almost_equal(dv1, dv2, 3)
            
        learner.lmbdaV = 0.5 
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
    
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += learner.derivativeViApprox(indPtr, colInds, U, V,  gp, gq, normGp, normGq, permutedRowInds, permutedColInds, i)
            dv1 /= numRuns
            dv2 = learner.derivativeVi(indPtr, colInds, U, V, gp, gq, i) 
            print(dv1, dv2)
            nptst.assert_array_almost_equal(dv1, dv2, 3)
            
        learner.numRowSamples = 10 
        numRuns = 1000
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += learner.derivativeViApprox(indPtr, colInds, U, V, gp, gq, normGp, normGq, permutedRowInds, permutedColInds, i)
            dv1 /= numRuns
            dv2 = learner.derivativeVi(indPtr, colInds, U, V, gp, gq, i)  
            print(dv1, dv2)
            nptst.assert_array_almost_equal(dv1, dv2, 3)

        maxLocalAuc.numRowSamples = m 
        maxLocalAuc.numAucSamples = 20 
        maxLocalAuc.lmbdaV = 0
        numRuns = 1000
        print("Final test")
        
        #for i in numpy.random.permutation(m)[0:numTests]: 
        for i in range(m): 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += learner.derivativeViApprox(indPtr, colInds, U, V, gp, gq, normGp, normGq, permutedRowInds, permutedColInds, i)
            dv1 /= numRuns
            #dv1 = learner.derivativeVi(indPtr, colInds, U, V, gp, gq, i) 
            dv2 = learner.derivativeVi(indPtr, colInds, U, V, gp, gq, i)   
                      
            
            print(i, dv1, dv2)
            nptst.assert_array_almost_equal(dv1, dv2, 3)


    @unittest.skip("")
    def testObjectiveApprox(self): 
        """
        We'll test the case in which we apprormate using a large number of samples 
        for the AUC and see if we get close to the exact objective 
        """
        m = 20 
        n = 30 
        k = 3 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, csarray=True)
        
        learner = MaxAUCLogistic(k)
        learner.normalise = False
        learner.lmbdaU = 0
        learner.lmbdaV = 0
        learner.rho = 1.0
        learner.numAucSamples = n
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)

        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)
        
        numRuns = 100 
        numTests = 5
        
        gi = numpy.random.rand(m)
        gi /= gi.sum()        
        gp = numpy.random.rand(n)
        gp /= gp.sum()        
        gq = numpy.random.rand(n)
        gq /= gq.sum()
        #gi = numpy.ones(m)
        #gp = numpy.ones(n)
        #gq = numpy.ones(n)

        #Let's compare against using the exact derivative 
        for i in range(numTests): 
            obj = 0

            for j in range(numRuns): 
                obj += learner.objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, gp, gq)
            obj /= numRuns

            obj2 = learner.objective(indPtr, colInds, indPtr, colInds, U, V, gp, gq)    
            self.assertAlmostEquals(obj, obj2, 2)
            
        learner.rho = 0.2

        for i in range(numTests): 
            obj = 0
            for j in range(numRuns): 
                obj += learner.objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, gp, gq)
            obj /= numRuns
            
            obj2 = learner.objective(indPtr, colInds, indPtr, colInds, U, V, gp, gq)    
            self.assertAlmostEquals(obj, obj2, 2)

        learner.lmbdaV = 0.2

        for i in range(numTests): 
            obj = 0
            for j in range(numRuns): 
                obj += learner.objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, gp, gq)
            obj /= numRuns
            
            obj2 = learner.objective(indPtr, colInds, indPtr, colInds, U, V, gp, gq)    
            self.assertAlmostEquals(obj, obj2, 2)
        
        #Check full and summary versions are the same 
        obj = learner.objective(indPtr, colInds, indPtr, colInds, U, V, gp, gq) 
        obj2 = learner.objective(indPtr, colInds, indPtr, colInds, U, V, gp, gq) 
        self.assertAlmostEquals(obj, obj2, 2)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()