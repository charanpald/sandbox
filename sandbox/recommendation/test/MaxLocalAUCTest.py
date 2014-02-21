
import sys
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
    def testLearnModel(self): 
        m = 50 
        n = 20 
        k = 5 
        numInds = 500
        X = SparseUtils.generateSparseLowRank((m, n), k, numInds)
        
        X = X/X
        
        lmbda = 0.00
        r = numpy.ones(m)*0.0
        eps = 0.05
        maxLocalAuc = MaxLocalAUC(lmbda, k, r, sigma=5.0, eps=eps)
        
        U, V = maxLocalAuc.learnModel(X)
        
        #print(U)
        #print(V)

    #@unittest.skip("")
    def testDerivativeU(self): 
        m = 10 
        n = 20 
        k = 2 
        numInds = 100
        X = SparseUtils.generateSparseLowRank((m, n), k, numInds)
        
        X = X/X
        
        
        for lmbda in [0.0, 0.01, 0.1]: 
            r = numpy.ones(m)*0.0
            maxLocalAuc = MaxLocalAUC(lmbda, k, r)
            maxLocalAuc.approxDerivative = False
            omegaList = SparseUtils.getOmegaList(X)
    
            U = numpy.random.rand(m, k)
            V = numpy.random.rand(n, k)
            rowInds, colInds = X.nonzero()
    
            deltaU = numpy.zeros(U.shape)
            for i in range(X.shape[0]): 
                deltaU[i, :] = maxLocalAuc.derivativeUi(X, U, V, omegaList, i, r)    
    
            #deltaU, inds = maxLocalAuc.derivativeU(X, U, V, omegaList)
            
            deltaU2 = numpy.zeros(U.shape)    
            
            eps = 0.0001        
            
            for i in range(m): 
                for j in range(k):
                    tempU = U.copy() 
                    tempU[i,j] += eps
                    obj1 = maxLocalAuc.objective(X, tempU, V, omegaList, r)
                    
                    tempU = U.copy() 
                    tempU[i,j] -= eps
                    obj2 = maxLocalAuc.objective(X, tempU, V, omegaList, r)
                    
                    deltaU2[i,j] = (obj1-obj2)/(2*eps)
                    
            #print(deltaU.T*10)
            #print(deltaU2.T*10)                      
            nptst.assert_almost_equal(deltaU, deltaU2, 2)

    #@unittest.skip("")
    def testDerivativeV(self): 
        m = 10 
        n = 20 
        k = 2 
        numInds = 100
        X = SparseUtils.generateSparseLowRank((m, n), k, numInds)
        
        X = X/X
        
        for lmbda in [0.0, 0.01, 0.1]: 
            r = numpy.ones(m)*0.0
            maxLocalAuc = MaxLocalAUC(lmbda, k, r)
            maxLocalAuc.approxDerivative = False
            omegaList = SparseUtils.getOmegaList(X)
    
            U = numpy.random.rand(m, k)
            V = numpy.random.rand(n, k)
            rowInds, colInds = X.nonzero()
    
            deltaV = numpy.zeros(V.shape)
            for i in range(X.shape[1]): 
                deltaV[i, :] = maxLocalAuc.derivativeVi(X, U, V, omegaList, i, r)    
    
            #deltaV, inds = maxLocalAuc.derivativeV(X, U, V, omegaList)
            
            deltaV2 = numpy.zeros(V.shape)    
            
            eps = 0.001        
            
            for i in range(n): 
                for j in range(k):
                    tempV = V.copy() 
                    tempV[i,j] += eps
                    obj1 = maxLocalAuc.objective(X, U, tempV, omegaList, r)
                    
                    tempV = V.copy() 
                    tempV[i,j] -= eps
                    obj2 = maxLocalAuc.objective(X, U, tempV, omegaList, r)
                    
                    deltaV2[i,j] = (obj1-obj2)/(2*eps)
             
            #print(deltaV.T*10)
            #print(deltaV2.T*10)                   
            nptst.assert_almost_equal(deltaV, deltaV2, 2)

    @unittest.skip("")
    def testModelSelect(self): 
        m = 10 
        n = 20 
        k = 2 
        numInds = 100
        X = SparseUtils.generateSparseLowRank((m, n), k, numInds)
        
        X = X/X
        
        r = numpy.ones(m)*0.0
        lmbda = 0.001
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(lmbda, k, r, eps=eps)
        
        maxLocalAuc.modelSelect(X)
            

    @unittest.skip("")
    def testLearningRateSelect(self): 
        m = 10 
        n = 20 
        k = 2 
        numInds = 100
        X = SparseUtils.generateSparseLowRank((m, n), k, numInds)
        
        X = X/X
        
        u= 0.1
        lmbda = 0.001
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(lmbda, k, u, eps=eps)
        maxLocalAuc.rate = "optimal"
        maxLocalAuc.maxIterations = 200
        
        maxLocalAuc.learningRateSelect(X)

    def testStr(self): 
        k=10
        u= 0.1
        lmbda = 0.001
        eps = 0.001
        maxLocalAuc = MaxLocalAUC(lmbda, k, u, eps=eps)    
        
        print(maxLocalAuc)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()