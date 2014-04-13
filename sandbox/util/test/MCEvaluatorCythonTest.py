import logging
import unittest
import numpy
import scipy.sparse 
import sklearn.metrics 
import numpy.testing as nptst 
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.Util import Util 

class  MCEvaluatorCythonTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(21)
        
        
    def testRecommendAtk(self): 
        m = 20 
        n = 50 
        r = 3 

        X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), r, 0.5, verbose=True)

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

            
if __name__ == '__main__':
    unittest.main()