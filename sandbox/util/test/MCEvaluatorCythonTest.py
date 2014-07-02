import logging
import unittest
import numpy
import scipy.sparse 
import sklearn.metrics 
import numpy.testing as nptst 
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.util.Util import Util 

class  MCEvaluatorCythonTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(21)
        
        
    def testRecommendAtk(self): 
        m = 20 
        n = 50 
        r = 3 

        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), r, 0.5, verbose=True)

        import sppy 
        X = sppy.csarray(X)  
        
        k = 10        
        
        X = numpy.zeros(X.shape)
        omegaList = []
        for i in range(m): 
            omegaList.append(numpy.random.permutation(n)[0:5])
            X[i, omegaList[i]] = 1
            
        X = sppy.csarray(X)            
        
        
        orderedItems = MCEvaluatorCython.recommendAtk(U, V, k, X)
        orderedItems2 = MCEvaluator.recommendAtk(U, V, k, omegaList=omegaList)
                
        nptst.assert_array_equal(orderedItems[orderedItems2!=-1], orderedItems2[orderedItems2!=-1])

        for i in range(m): 
            items = numpy.intersect1d(omegaList[i], orderedItems[i, :])
            self.assertEquals(items.shape[0], 0)
            
            #items = numpy.union1d(omegaList[i], orderedItems[i, :])
            #items = numpy.intersect1d(items, orderedItems2[i, :])
            #nptst.assert_array_equal(items, numpy.sort(orderedItems2[i, :]))
            
        #Now let's have an all zeros X 
        X = sppy.csarray(X.shape)
        orderedItems = MCEvaluatorCython.recommendAtk(U, V, k, X)
        orderedItems2 = MCEvaluator.recommendAtk(U, V, k) 
        
        nptst.assert_array_equal(orderedItems, orderedItems2)

            
    def testReciprocalRankAtk(self): 
        m = 20 
        n = 50 
        r = 3 
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), r, 0.5, verbose=True, csarray=True)
        
        k = 5
        orderedItems = numpy.random.randint(0, n, m*k)
        orderedItems = numpy.reshape(orderedItems, (m, k))
        orderedItems = numpy.array(orderedItems, numpy.int32)
        
        (indPtr, colInds) = X.nonzeroRowsPtr()
        rrs = MCEvaluatorCython.reciprocalRankAtk(indPtr, colInds, orderedItems)
        
        rrs2 = numpy.zeros(m)
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            for j in range(k): 
                if orderedItems[i, j] in omegai: 
                    rrs2[i] = 1/float(1+j)
                    break 
        
        nptst.assert_array_equal(rrs, rrs2)
        
        #Test case where no items are in ranking 
        orderedItems = numpy.ones((m, k), numpy.int32) * (n+1)
        rrs = MCEvaluatorCython.reciprocalRankAtk(indPtr, colInds, orderedItems)
        nptst.assert_array_equal(rrs, numpy.zeros(m))
        
        #Now, make all items rank 2
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            orderedItems[i, 1] = omegai[0]
        
        rrs = MCEvaluatorCython.reciprocalRankAtk(indPtr, colInds, orderedItems)
        nptst.assert_array_equal(rrs, numpy.ones(m)*0.5)     
            
    def testStratifiedRecallAtk(self): 
        m = 20 
        n = 50 
        r = 3     
        alpha = 1
        
        X, U, V = SparseUtilsCython.generateSparseBinaryMatrixPL((m,n), r, density=0.2, alpha=alpha, csarray=True)
        
        itemCounts = numpy.array(X.sum(0)+1, numpy.int32) 
        
        (indPtr, colInds) = X.nonzeroRowsPtr()
        
        k = 5
        orderedItems = numpy.random.randint(0, n, m*k)
        orderedItems = numpy.reshape(orderedItems, (m, k))
        orderedItems = numpy.array(orderedItems, numpy.int32)        
        beta = 0.5
        
        recalls, denominators = MCEvaluatorCython.stratifiedRecallAtk(indPtr, colInds, orderedItems, itemCounts, beta)
        
        
        recalls2 = numpy.zeros(m)        
            
        #Now compute recalls from scratch 
        for i in range(m):
            omegai = colInds[indPtr[i]:indPtr[i+1]]            
            
            numerator = 0 
            for j in range(k):
                if orderedItems[i, j] in omegai: 
                    numerator += 1/itemCounts[orderedItems[i, j]]**beta
            
            denominator = 0

            for j in omegai: 
                denominator += 1/itemCounts[j]**beta
                
            recalls2[i] = numerator/denominator
            
        nptst.assert_array_equal(recalls, recalls2)
                                
                
        #Now try to match with normal recall 
        itemCounts = numpy.ones(n, numpy.int32)
        recalls, denominators = MCEvaluatorCython.stratifiedRecallAtk(indPtr, colInds, orderedItems, itemCounts, beta)
        recalls2 = MCEvaluatorCython.recallAtk(indPtr, colInds, orderedItems)
        
        nptst.assert_array_equal(recalls, recalls2)
                    
            
if __name__ == '__main__':
    unittest.main()