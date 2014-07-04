
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
        numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
        
        numpy.seterr(all="raise")
        numpy.random.seed(21)


    @unittest.skip("")
    def testDerivativeU(self): 
        m = 10 
        n = 20 
        nnzPerRow = 5 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), nnzPerRow, csarray=True)
        
        k = 5
        u = 0.1
        w = 1-u
        eps = 0.05
        maxLocalAuc = MaxLocalAUC(k, w, alpha=1.0, eps=eps)
        maxLocalAuc.normalise = False
        maxLocalAuc.lmbda = 0
        maxLocalAuc.rho = 1.0
        maxLocalAuc.numAucSamples = 100

        numRuns = 20
        r = numpy.zeros(m)
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
                deltaU[i, :] = derivativeUi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)      
    
            deltaU2 = numpy.zeros(U.shape) 
            eps = 10**-6         
            
            for i in range(m): 
                for j in range(k):
                    tempU = U.copy() 
                    tempU[i,j] += eps
                    obj1 = objective(indPtr, colInds, indPtr, colInds, tempU, V, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
                    tempU = U.copy() 
                    tempU[i,j] -= eps
                    obj2 = objective(indPtr, colInds, indPtr, colInds, tempU, V, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
                    deltaU2[i,j] = (obj1-obj2)/(2*eps)
    
                #deltaU2[i,:] = deltaU2[i,:]/numpy.linalg.norm(deltaU2[i,:])
            
            nptst.assert_almost_equal(deltaU, deltaU2, 3)
        
        #Try r != 0 and rho > 0
        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)
            r = numpy.random.rand(m)
            maxLocalAuc.rho = 0.1
            
            deltaU = numpy.zeros(U.shape)
            for i in range(X.shape[0]): 
                deltaU[i, :] = derivativeUi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)
            
            deltaU2 = numpy.zeros(U.shape) 
            eps = 10**-9        
            
            for i in range(m): 
                for j in range(k):
                    tempU = U.copy() 
                    tempU[i,j] += eps
                    obj1 = objective(indPtr, colInds, indPtr, colInds, tempU, V, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
                    tempU = U.copy() 
                    tempU[i,j] -= eps
                    obj2 = objective(indPtr, colInds, indPtr, colInds, tempU, V, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
                    deltaU2[i,j] = (obj1-obj2)/(2*eps)
                                
            nptst.assert_almost_equal(deltaU, deltaU2, 3)
        
        #Try lmbda > 0
        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)
            maxLocalAuc.lmbda = 0.1
            
            deltaU = numpy.zeros(U.shape)
            for i in range(X.shape[0]): 
                deltaU[i, :] = derivativeUi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False) 
            
            deltaU2 = numpy.zeros(U.shape) 
            eps = 10**-9        
            
            for i in range(m): 
                for j in range(k):
                    tempU = U.copy() 
                    tempU[i,j] += eps
                    obj1 = objective(indPtr, colInds, indPtr, colInds, tempU, V, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
                    tempU = U.copy() 
                    tempU[i,j] -= eps
                    obj2 = objective(indPtr, colInds, indPtr, colInds, tempU, V, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
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
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(k, w, eps=eps)
        maxLocalAuc.numAucSamples = n**2
        maxLocalAuc.numRowSamples = m 
        maxLocalAuc.lmbda = 0
        maxLocalAuc.rho = 0 

        
        U = numpy.random.rand(X.shape[0], k)
        V = numpy.random.rand(X.shape[1], k)

        gi = numpy.random.rand(m)
        gi /= gi.sum()        
        gp = numpy.random.rand(n)
        gp /= gp.sum()        
        gq = numpy.random.rand(n)
        gq /= gq.sum()     

        
        numRuns = 200 
        numTests = 5
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
        r = numpy.zeros(m)
        colIndsProbs = numpy.ones(colInds.shape[0])
        c = numpy.ones(n)
        
        for i in range(m): 
            colIndsProbs[indPtr[i]:indPtr[i+1]] /= colIndsProbs[indPtr[i]:indPtr[i+1]].sum()
            colIndsProbs[indPtr[i]:indPtr[i+1]] = numpy.cumsum(colIndsProbs[indPtr[i]:indPtr[i+1]])

        #Test with small number of AUC samples, but normalise 
        maxLocalAuc.numAucSamples = 30
        numRuns = 1000
        
        for i in numpy.random.permutation(m)[0:numTests]:  
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(indPtr, colInds, colIndsProbs, U, V, r, gi, gp, gq, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False) 
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
                du1 += derivativeUiApprox(indPtr, colInds, colIndsProbs, U, V, r, gi, gp, gq, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)   
            
            print(du1/numpy.linalg.norm(du1), du2/numpy.linalg.norm(du2))
            nptst.assert_array_almost_equal(du1, du2, 2)
            
            
        maxLocalAuc.rho = 0.1

        for i in numpy.random.permutation(m)[0:numTests]:  
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(indPtr, colInds, colIndsProbs, U, V, r, gi, gp, gq, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(du1, du2, 2)
            print(du1, du2)
            
        maxLocalAuc.lmbda = 0.5 
        
        for i in numpy.random.permutation(m)[0:numTests]:  
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(indPtr, colInds, colIndsProbs, U, V, r, gi, gp, gq, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(du1, du2, 2)
            print(du1, du2)
            
        #Test varying c 
        c = numpy.random.rand(n)
        
        for i in numpy.random.permutation(m)[0:numTests]:  
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            du1 = numpy.zeros(k)
            for j in range(numRuns): 
                du1 += derivativeUiApprox(indPtr, colInds, colIndsProbs, U, V, r, gi, gp, gq, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
            du1 /= numRuns
            du2 = derivativeUi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(du1, du2, 2)
            print(du1, du2)
            
            

    @unittest.skip("")
    def testDerivativeV(self): 
        m = 10 
        n = 20 
        nnzPerRow = 5 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), nnzPerRow, csarray=True)
        
        k = 5
        u = 0.1
        w = 1-u
        eps = 0.05
        maxLocalAuc = MaxLocalAUC(k, w, alpha=1.0, eps=eps)
        maxLocalAuc.normalise = False
        maxLocalAuc.lmbda = 0
        maxLocalAuc.rho = 0
        maxLocalAuc.numAucSamples = 100

        r = numpy.zeros(m)

        numRuns = 20
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
        
        gi = numpy.random.rand(m)
        gi /= gi.sum()        
        gp = numpy.random.rand(n)
        gp /= gp.sum()        
        gq = numpy.random.rand(n)
        gq /= gq.sum()            
        

        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)            
            
            deltaV = numpy.zeros(V.shape)
            for j in range(n): 
                deltaV[j, :] = derivativeVi(indPtr, colInds, U, V, r, gi, gp, gq, j, maxLocalAuc.rho, False)   
            
            deltaV2 = numpy.zeros(V.shape)    
            
            eps = 0.00001        
            
            for i in range(n): 
                for j in range(k):
                    tempV = V.copy() 
                    tempV[i,j] += eps
                    obj1 = objective(indPtr, colInds, indPtr, colInds, U, tempV, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
                    tempV = V.copy() 
                    tempV[i,j] -= eps
                    obj2 = objective(indPtr, colInds, indPtr, colInds, U, tempV, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
                    deltaV2[i,j] = (obj1-obj2)/(2*eps)
                #deltaV2[i,:] = deltaV2[i,:]/numpy.linalg.norm(deltaV2[i,:])                   
                        

            nptst.assert_almost_equal(deltaV, deltaV2, 3)

        #Try r != 0 and rho > 0
        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)   
            r = numpy.random.rand(m)
            maxLocalAuc.rho = 1.0    
            
            deltaV = numpy.zeros(V.shape)
            for j in range(n): 
                deltaV[j, :] = derivativeVi(indPtr, colInds, U, V, r, gi, gp, gq, j, maxLocalAuc.rho, False)    
            
            deltaV2 = numpy.zeros(V.shape)
            
            for i in range(n): 
                for j in range(k):
                    tempV = V.copy() 
                    tempV[i,j] += eps
                    obj1 = objective(indPtr, colInds, indPtr, colInds, U, tempV, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
                    tempV = V.copy() 
                    tempV[i,j] -= eps
                    obj2 = objective(indPtr, colInds, indPtr, colInds, U, tempV, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
                    deltaV2[i,j] = (obj1-obj2)/(2*eps)
                #deltaV2[i,:] = deltaV2[i,:]/numpy.linalg.norm(deltaV2[i,:])
                           
            nptst.assert_almost_equal(deltaV, deltaV2, 3)
        
        
        #Try r != 0 and rho > 0
        for s in range(numRuns):
            U = numpy.random.randn(m, k)
            V = numpy.random.randn(n, k)   
            maxLocalAuc.lmbda = 0.1    
            
            deltaV = numpy.zeros(V.shape)
            for j in range(n): 
                deltaV[j, :] = derivativeVi(indPtr, colInds, U, V, r, gi, gp, gq, j, maxLocalAuc.rho, False)
            
            deltaV2 = numpy.zeros(V.shape)
            
            for i in range(n): 
                for j in range(k):
                    tempV = V.copy() 
                    tempV[i,j] += eps
                    obj1 = objective(indPtr, colInds, indPtr, colInds, U, tempV, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
                    tempV = V.copy() 
                    tempV[i,j] -= eps
                    obj2 = objective(indPtr, colInds, indPtr, colInds, U, tempV, r, gi, gp, gq, maxLocalAuc.rho, False)
                    
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
        
        w = 0.1
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(k, w, eps=eps)
        maxLocalAuc.numAucSamples = n**2
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
        
        gi = numpy.random.rand(m)
        gi /= gi.sum()        
        gp = numpy.random.rand(n)
        gp /= gp.sum()        
        gq = numpy.random.rand(n)
        gq /= gq.sum()   
        gi = numpy.ones(m)
        gp = numpy.ones(n)
        gq = numpy.ones(n)         
        
        numRuns = 500 
        numTests = 5

        #Let's compare against using the exact derivative 
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples,  maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)   
            
            
            dv3 = numpy.zeros(k)
            for j in range(k): 
                eps = 10**-6
                tempV = V.copy() 
                tempV[i,j] += eps
                obj1 = objective(indPtr, colInds, indPtr, colInds, U, tempV, r, gi, gp, gq, maxLocalAuc.rho, False)
                
                tempV = V.copy() 
                tempV[i,j] -= eps
                obj2 = objective(indPtr, colInds, indPtr, colInds, U, tempV, r, gi, gp, gq, maxLocalAuc.rho, False)
                
                dv3[j] = (obj1-obj2)/(2*eps)            
            
            print(dv1, dv2, dv3)
            print(dv1/numpy.linalg.norm(dv1), dv2/numpy.linalg.norm(dv2), dv3/numpy.linalg.norm(dv3))
            
            nptst.assert_array_almost_equal(dv1, dv2, 2)
            
        maxLocalAuc.rho = 0.2

        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(dv1, dv2, 3)
            
        maxLocalAuc.lmbda = 0.5 
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
    
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples,  maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(dv1, dv2, 2)
            
        maxLocalAuc.numRowSamples = 10 
        numRuns = 5000
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)   
            nptst.assert_array_almost_equal(dv1, dv2, 3)

        maxLocalAuc.numRowSamples = m 
        maxLocalAuc.numAucSamples = 10 
        numRuns = 5000
        
        for i in numpy.random.permutation(m)[0:numTests]: 
            U = numpy.random.rand(X.shape[0], k)
            V = numpy.random.rand(X.shape[1], k)            
            
            dv1 = numpy.zeros(k)
            for j in range(numRuns): 
                dv1 += derivativeViApprox(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.numRowSamples, maxLocalAuc.numAucSamples,  maxLocalAuc.rho, False)
            dv1 /= numRuns
            dv2 = derivativeVi(indPtr, colInds, U, V, r, gi, gp, gq, i, maxLocalAuc.rho, False)   
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
        
        w = 0.1
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(k, w, eps=eps)
        maxLocalAuc.numAucSamples = n*2
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
                obj += objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
            obj /= numRuns
            
            obj2 = objective(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.rho, False)    
            self.assertAlmostEquals(obj, obj2, 2)
            
        maxLocalAuc.rho = 0.2

        for i in range(numTests): 
            obj = 0
            for j in range(numRuns): 
                obj += objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
            obj /= numRuns
            
            obj2 = objective(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.rho, False)    
            self.assertAlmostEquals(obj, obj2, 2)

        maxLocalAuc.lmbda = 0.2

        for i in range(numTests): 
            obj = 0
            for j in range(numRuns): 
                obj += objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
            obj /= numRuns
            
            obj2 = objective(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.rho, False)    
            self.assertAlmostEquals(obj, obj2, 2)
        
        #Check full and summary versions are the same 
        obj = objective(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.rho, True) 
        obj = obj.sum()
        obj2 = objective(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.rho, False) 
        self.assertAlmostEquals(obj, obj2, 2)
        
        
        U = numpy.zeros((X.shape[0], k))
        V = numpy.zeros((X.shape[1], k))
        r = numpy.zeros(X.shape[0])
        
        maxLocalAuc.rho = 0 
        obj = 0
        for j in range(numRuns): 
            obj += objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
        obj /= numRuns
        
        obj2 = objective(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.rho, False)
        
        #self.assertEquals(obj, 0.5)
        #self.assertEquals(obj2, 0.5)
        
        maxLocalAuc.rho = 1
        r = numpy.ones(X.shape[0])
        obj = 0
        for j in range(numRuns): 
            obj += objectiveApprox(indPtr, colInds, indPtr, colInds, U, V, r, gi, gp, gq, maxLocalAuc.numAucSamples, maxLocalAuc.rho, False)
        obj /= numRuns        
        
        #self.assertAlmostEquals(obj, 2.0)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()