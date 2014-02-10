import logging
import unittest
import numpy
import scipy.sparse 
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.SparseUtils import SparseUtils

class  MCEvaluatorTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(21)
    
    def testMeanSqError(self): 
        numExamples = 10
        testX = scipy.sparse.rand(numExamples, numExamples)
        testX = testX.tocsr()
        
        predX = testX.copy() 
        error = MCEvaluator.meanSqError(testX, predX)
        self.assertEquals(error, 0.0)
        
        testX = numpy.random.rand(numExamples, numExamples)
        predX = testX + numpy.random.rand(numExamples, numExamples)*0.5 
        
        error2 = ((testX-predX)**2).sum()/(numExamples**2)
        error = MCEvaluator.meanSqError(scipy.sparse.csr_matrix(testX), scipy.sparse.csr_matrix(predX)) 
        
        self.assertEquals(error, error2)
        
    def testPrecisionAtK(self): 
        m = 10 
        n = 5 
        r = 3 
        k = m*n
        
        X, U, s, V = SparseUtils.generateSparseLowRank((m,n), r, k, verbose=True)
        mean = X.data.mean()
        X.data[X.data <= mean] = 0
        X.data[X.data > mean] = 1
        
        import sppy 
        X = sppy.csarray(X)
        
        print(MCEvaluator.precisionAtK(X, U, V, 4))
        
if __name__ == '__main__':
    unittest.main()

