import os
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC 
from sandbox.recommendation.MaxLocalAUCCython import objectiveApprox, objective
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
        numpy.random.seed(22)
    
    @unittest.skip("")
    def testLearnModel(self): 
        m = 50 
        n = 20 
        k = 5 
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, csarray=True)

        
        u = 0.1
        w = 1-u
        eps = 0.05
        maxLocalAuc = MaxLocalAUC(k, w, alpha=5.0, eps=eps)
        
        U, V = maxLocalAuc.learnModel(X)
        
        
        maxLocalAuc.stochastic = True 
        U, V = maxLocalAuc.learnModel(X)
        #print(U)
        #print(V)

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
        maxLocalAuc.C = 0
        maxLocalAuc.numAucSamples = 100

        U = numpy.random.rand(m, k)
        V = numpy.random.rand(n, k)
        xi = numpy.zeros(m)
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)

        deltaU = numpy.zeros(U.shape)
        for i in range(X.shape[0]): 
            deltaU[i, :] = maxLocalAuc.derivativeUi(indPtr, colInds, U, V, xi, i)    

        deltaU2 = numpy.zeros(U.shape) 
        eps = 10**-12         
        
        for i in range(m): 
            for j in range(k):
                tempU = U.copy() 
                tempU[i,j] += eps
                obj1 = objective(indPtr, colInds, indPtr, colInds, tempU, V, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                tempU = U.copy() 
                tempU[i,j] -= eps
                obj2 = objective(indPtr, colInds, indPtr, colInds, tempU, V, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                deltaU2[i,j] = (obj1-obj2)/(2*eps)

            #deltaU2[i,:] = deltaU2[i,:]/numpy.linalg.norm(deltaU2[i,:])

        nptst.assert_almost_equal(deltaU, deltaU2, 2)
        
        #Try xi != 0 and C > 0
        xi = numpy.random.rand(m)
        maxLocalAuc.C = 0.1
        
        deltaU = numpy.zeros(U.shape)
        for i in range(X.shape[0]): 
            deltaU[i, :] = maxLocalAuc.derivativeUi(indPtr, colInds, U, V, xi, i) 
        
        deltaU2 = numpy.zeros(U.shape) 
        eps = 10**-9        
        
        for i in range(m): 
            for j in range(k):
                tempU = U.copy() 
                tempU[i,j] += eps
                obj1 = objective(indPtr, colInds, indPtr, colInds, tempU, V, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                tempU = U.copy() 
                tempU[i,j] -= eps
                obj2 = objective(indPtr, colInds, indPtr, colInds, tempU, V, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                deltaU2[i,j] = (obj1-obj2)/(2*eps)
                            
        nptst.assert_almost_equal(deltaU, deltaU2, 2)
        
        #Try lmbda > 0
        maxLocalAuc.lmbda = 0.1
        
        deltaU = numpy.zeros(U.shape)
        for i in range(X.shape[0]): 
            deltaU[i, :] = maxLocalAuc.derivativeUi(indPtr, colInds, U, V, xi, i) 
        
        deltaU2 = numpy.zeros(U.shape) 
        eps = 10**-9        
        
        for i in range(m): 
            for j in range(k):
                tempU = U.copy() 
                tempU[i,j] += eps
                obj1 = objective(indPtr, colInds, indPtr, colInds, tempU, V, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                tempU = U.copy() 
                tempU[i,j] -= eps
                obj2 = objective(indPtr, colInds, indPtr, colInds, tempU, V, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                deltaU2[i,j] = (obj1-obj2)/(2*eps)
                            
        nptst.assert_almost_equal(deltaU, deltaU2, 2)

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
        maxLocalAuc.C = 0
        maxLocalAuc.numAucSamples = 100

        U = numpy.random.rand(m, k)
        V = numpy.random.rand(n, k)
        xi = numpy.zeros(m)
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)

        deltaV = numpy.zeros(V.shape)
        for j in range(n): 
            deltaV[j, :] = maxLocalAuc.derivativeVi(indPtr, colInds, U, V, xi, j)    
        
        deltaV2 = numpy.zeros(V.shape)    
        
        eps = 0.001        
        
        for i in range(n): 
            for j in range(k):
                tempV = V.copy() 
                tempV[i,j] += eps
                obj1 = objective(indPtr, colInds, indPtr, colInds, U, tempV, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                tempV = V.copy() 
                tempV[i,j] -= eps
                obj2 = objective(indPtr, colInds, indPtr, colInds, U, tempV, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                deltaV2[i,j] = (obj1-obj2)/(2*eps)
            #deltaV2[i,:] = deltaV2[i,:]/numpy.linalg.norm(deltaV2[i,:])
                       
        nptst.assert_almost_equal(deltaV, deltaV2, 2)

        #Try xi != 0 and C > 0
        xi = numpy.random.rand(m)
        maxLocalAuc.C = 1.0    
        
        deltaV = numpy.zeros(V.shape)
        for j in range(n): 
            deltaV[j, :] = maxLocalAuc.derivativeVi(indPtr, colInds, U, V, xi, j)    
        
        deltaV2 = numpy.zeros(V.shape)
        
        for i in range(n): 
            for j in range(k):
                tempV = V.copy() 
                tempV[i,j] += eps
                obj1 = objective(indPtr, colInds, indPtr, colInds, U, tempV, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                tempV = V.copy() 
                tempV[i,j] -= eps
                obj2 = objective(indPtr, colInds, indPtr, colInds, U, tempV, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                deltaV2[i,j] = (obj1-obj2)/(2*eps)
            #deltaV2[i,:] = deltaV2[i,:]/numpy.linalg.norm(deltaV2[i,:])
                       
        nptst.assert_almost_equal(deltaV, deltaV2, 2)
        
        
        #Try xi != 0 and C > 0
        maxLocalAuc.lmbda = 0.1    
        
        deltaV = numpy.zeros(V.shape)
        for j in range(n): 
            deltaV[j, :] = maxLocalAuc.derivativeVi(indPtr, colInds, U, V, xi, j)    
        
        deltaV2 = numpy.zeros(V.shape)
        
        for i in range(n): 
            for j in range(k):
                tempV = V.copy() 
                tempV[i,j] += eps
                obj1 = objective(indPtr, colInds, indPtr, colInds, U, tempV, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                tempV = V.copy() 
                tempV[i,j] -= eps
                obj2 = objective(indPtr, colInds, indPtr, colInds, U, tempV, xi, maxLocalAuc.lmbda, maxLocalAuc.C, False)
                
                deltaV2[i,j] = (obj1-obj2)/(2*eps)
            #deltaV2[i,:] = deltaV2[i,:]/numpy.linalg.norm(deltaV2[i,:])
                       
        nptst.assert_almost_equal(deltaV, deltaV2, 2)

    @unittest.skip("")
    def testModelSelect(self): 
        m = 10 
        n = 20 
        k = 5 
        
        u = 0.5
        w = 1-u
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, w, csarray=True)
        
        os.system('taskset -p 0xffffffff %d' % os.getpid())
        
        eps = 0.001
        k = 5
        maxLocalAuc = MaxLocalAUC(k, w, eps=eps, stochastic=True)
        maxLocalAuc.maxIterations = 5
        #maxLocalAuc.numProcesses = 1
        maxLocalAuc.recordStep = 1
        maxLocalAuc.validationSize = 3
        
        maxLocalAuc.modelSelect(X)
            

    @unittest.skip("")
    def testLearningRateSelect(self): 
        m = 10 
        n = 20 
        k = 5 
        
        u = 0.5
        w = 1-u
        X = SparseUtils.generateSparseBinaryMatrix((m, n), k, w, csarray=True)
        
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(k, u, eps=eps)
        maxLocalAuc.rate = "optimal"
        maxLocalAuc.maxIterations = 5
        
        maxLocalAuc.learningRateSelect(X)

    def testStr(self): 
        k=10
        u= 0.1
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(k, u, eps=eps)    
        
        print(maxLocalAuc)

    def testCopy(self): 
        u= 0.1
        eps = 0.001
        k = 10 
        maxLocalAuc = MaxLocalAUC(k, u, alpha=5.0, eps=eps)
        maxLocalAuc.copy()

    def testOmegaProbsUniform(self):
        m = 10 
        n = 20
        p  = 5
        X = SparseUtils.generateSparseBinaryMatrix((m, n), p, csarray=True)
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)

        u= 0.1
        eps = 0.001
        k = 10 
        maxLocalAuc = MaxLocalAUC(k, u, alpha=5.0, eps=eps)
        
        U, V = maxLocalAuc.initUV(X)
        
        colIndsCumProbs = maxLocalAuc.omegaProbsUniform(indPtr, colInds, U, V)
        
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            omegaiProbs = colIndsCumProbs[indPtr[i]:indPtr[i+1]]
            self.assertAlmostEquals(omegaiProbs[-1], 1)
            
            probs = numpy.ones(omegai.shape[0])
            probs /= probs.sum()
            probs = numpy.cumsum(probs)
            nptst.assert_array_almost_equal(probs, omegaiProbs)
        

    def testOmegaProbsTopZ(self):
        m = 10 
        n = 20
        u = 0.5
        w = 1-u
        p  = 5
        X = SparseUtils.generateSparseBinaryMatrix((m, n), p, w, csarray=True)
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
        
        X2 = numpy.zeros((m, n))
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            X2[i, omegai] = 1
            
        nptst.assert_array_equal(X.toarray(), X2)

        eps = 0.001
        k = 10 
        maxLocalAuc = MaxLocalAUC(k, w, alpha=5.0, eps=eps)
        
        U, V = maxLocalAuc.initUV(X)
        
        colIndsCumProbs = maxLocalAuc.omegaProbsTopZ(indPtr, colInds, U, V)
        
        Z = U.dot(V.T)
        
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            omegaiProbs = colIndsCumProbs[indPtr[i]:indPtr[i+1]]
            self.assertAlmostEquals(omegaiProbs[-1], 1)
            
            vals = Z[i, omegai]
            
            sortedVals = numpy.flipud(numpy.sort(vals))
            ri = sortedVals[min(maxLocalAuc.z, sortedVals.shape[0])-1]
            probs = numpy.zeros(omegai.shape[0])
            probs[vals >= ri] = 1
            probs /= probs.sum()
            
            probs = numpy.cumsum(probs)
            nptst.assert_array_almost_equal(probs, omegaiProbs)


    def testOmegaProbsRank(self):
        m = 10 
        n = 20
        p  = 5
        X = SparseUtils.generateSparseBinaryMatrix((m, n), p, csarray=True)
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)

        u= 0.1
        eps = 0.001
        k = 10 
        maxLocalAuc = MaxLocalAUC(k, u, alpha=5.0, eps=eps)
        
        U, V = maxLocalAuc.initUV(X)
        
        colIndsCumProbs = maxLocalAuc.omegaProbsRank(indPtr, colInds, U, V)
        
        Z = U.dot(V.T)
        
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            omegaiProbs = colIndsCumProbs[indPtr[i]:indPtr[i+1]]
            self.assertAlmostEquals(omegaiProbs[-1], 1)
            
            vals = Z[i, omegai]
            inds = numpy.argsort(numpy.argsort((vals)))+1

            probs = inds/float(inds.sum())
            probs = numpy.cumsum(probs)
            nptst.assert_array_almost_equal(probs, omegaiProbs)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()