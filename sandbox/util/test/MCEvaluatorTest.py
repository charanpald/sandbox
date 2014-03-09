import logging
import unittest
import numpy
import scipy.sparse 
import sklearn.metrics 
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

        X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), r, 0.5, verbose=True)

        import sppy 
        X = sppy.csarray(X)
        
        #print(MCEvaluator.precisionAtK(X, U*s, V, 2))
        
        orderedItems = MCEvaluator.recommendAtk(U, V, n)
        self.assertAlmostEquals(MCEvaluator.precisionAtK(X, orderedItems, n), X.nnz/float(m*n))
        
        k = 2
        orderedItems = MCEvaluator.recommendAtk(U*s, V, k)
        precision, scoreInds = MCEvaluator.precisionAtK(X, orderedItems, k, verbose=True)
        
        precisions = numpy.zeros(m)
        for i in range(m): 
            nonzeroRow = X.toarray()[i, :].nonzero()[0]

            precisions[i] = numpy.intersect1d(scoreInds[i, :], nonzeroRow).shape[0]/float(k)  
        
        self.assertEquals(precision, precisions.mean())
        
        #Now try random U and V 
        U = numpy.random.rand(m, 3)
        V = numpy.random.rand(m, 3)
        
        orderedItems = MCEvaluator.recommendAtk(U*s, V, k)
        precision, scoreInds = MCEvaluator.precisionAtK(X, orderedItems, k, verbose=True)
        
        precisions = numpy.zeros(m)
        for i in range(m): 
            nonzeroRow = X.toarray()[i, :].nonzero()[0]

            precisions[i] = numpy.intersect1d(scoreInds[i, :], nonzeroRow).shape[0]/float(k)  
        
        self.assertEquals(precision, precisions.mean())
        
    def testRecallAtK(self): 
        m = 10 
        n = 5 
        r = 3 

        X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), r, 0.5, verbose=True)

        import sppy 
        X = sppy.csarray(X)
        

        orderedItems = MCEvaluator.recommendAtk(U, V, n)
        self.assertAlmostEquals(MCEvaluator.recallAtK(X, orderedItems, n), 1.0)
        
        k = 2
        orderedItems = MCEvaluator.recommendAtk(U*s, V, k)
        recall, scoreInds = MCEvaluator.recallAtK(X, orderedItems, k, verbose=True)
        
        recalls = numpy.zeros(m)
        for i in range(m): 
            nonzeroRow = X.toarray()[i, :].nonzero()[0]

            recalls[i] = numpy.intersect1d(scoreInds[i, :], nonzeroRow).shape[0]/float(nonzeroRow.shape[0])
        
        self.assertEquals(recall, recalls.mean())
        
        #Now try random U and V 
        U = numpy.random.rand(m, 3)
        V = numpy.random.rand(m, 3)
        
        orderedItems = MCEvaluator.recommendAtk(U, V, k)
        recall, scoreInds = MCEvaluator.recallAtK(X, orderedItems, k, verbose=True)
        
        recalls = numpy.zeros(m)
        for i in range(m): 
            nonzeroRow = X.toarray()[i, :].nonzero()[0]

            recalls[i] = numpy.intersect1d(scoreInds[i, :], nonzeroRow).shape[0]/float(nonzeroRow.shape[0])    
        
        self.assertEquals(recall, recalls.mean())  
          
    def testLocalAUC(self): 
        m = 10 
        n = 20 
        k = 2 
        numInds = 100
        X, U, s, V = SparseUtils.generateSparseLowRank((m, n), k, numInds, verbose=True)
        
        X = X/X
        Z = U.dot(V.T)

        
        localAuc = numpy.zeros(m)
        
        for i in range(m): 
            localAuc[i] = sklearn.metrics.roc_auc_score(numpy.ravel(X[i, :].todense()), Z[i, :])
                    
        localAuc = localAuc.mean()
        
        u = 1.0
        localAuc2 = MCEvaluator.localAUC(X, U, V, u)

        self.assertEquals(localAuc, localAuc2)
        
        #Now try a large r 
        u =0

        localAuc2 = MCEvaluator.localAUC(X, U, V, u)
        self.assertEquals(localAuc2, 0)
        
    def testLocalAucApprox(self): 
        m = 100 
        n = 200 
        k = 2 
        numInds = 100
        X, U, s, V = SparseUtils.generateSparseLowRank((m, n), k, numInds, verbose=True)
        
        X = X/X
        Z = U.dot(V.T)

        u = 1.0
        
        
        localAuc = MCEvaluator.localAUC(X, U, V, u)
        
        samples = numpy.arange(50, 200, 10)
        
        for i, sampleSize in enumerate(samples): 
            numAucSamples = sampleSize
            localAuc2 = MCEvaluator.localAUCApprox(X, U, V, u, numAucSamples)

            self.assertAlmostEqual(localAuc2, localAuc, 1)        
        
if __name__ == '__main__':
    unittest.main()

