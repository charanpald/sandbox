import logging
import unittest
import numpy
import scipy.sparse 
import sklearn.metrics 
import numpy.testing as nptst 
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.Util import Util 

class  MCEvaluatorTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(21)
        numpy.set_printoptions(suppress=True, precision=3)
    
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

    def testRecommendAtk(self): 
        m = 20 
        n = 50 
        r = 3 

        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), r, 0.5, verbose=True)

        import sppy 
        X = sppy.csarray(X)  
        
        k = 10        
        orderedItems, scores = MCEvaluator.recommendAtk(U, V, k, verbose=True)
        
        #Now do it manually 
        Z = U.dot(V.T)
        
        orderedItems2 = Util.argmaxN(Z, k)
        scores2 = numpy.fliplr(numpy.sort(Z, 1))[:, 0:k]
        
        nptst.assert_array_equal(orderedItems, orderedItems2)
        nptst.assert_array_equal(scores, scores2)
        
        
        #Test case where we have a set of training indices to remove 
        #Let's create a random omegaList 
        omegaList = []
        for i in range(m): 
            omegaList.append(numpy.random.permutation(n)[0:5])
        
        
        orderedItems = MCEvaluator.recommendAtk(U, V, k, omegaList=omegaList)
        orderedItems2 = MCEvaluator.recommendAtk(U, V, k)
        
        #print(omegaList)
        #print(orderedItems)
        #print(orderedItems2)
        
        for i in range(m): 
            items = numpy.intersect1d(omegaList[i], orderedItems[i, :])
            self.assertEquals(items.shape[0], 0)
            
            items = numpy.union1d(omegaList[i], orderedItems[i, :])
            items = numpy.intersect1d(items, orderedItems2[i, :])
            nptst.assert_array_equal(items, numpy.sort(orderedItems2[i, :]))
        
        
    def testPrecisionAtK(self): 
        m = 10 
        n = 5 
        r = 3 

        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), r, 0.5, verbose=True)

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
        
        self.assertEquals(precision.mean(), precisions.mean())
        
        #Now try random U and V 
        U = numpy.random.rand(m, 3)
        V = numpy.random.rand(m, 3)
        
        orderedItems = MCEvaluator.recommendAtk(U*s, V, k)
        precision, scoreInds = MCEvaluator.precisionAtK(X, orderedItems, k, verbose=True)
        
        precisions = numpy.zeros(m)
        for i in range(m): 
            nonzeroRow = X.toarray()[i, :].nonzero()[0]

            precisions[i] = numpy.intersect1d(scoreInds[i, :], nonzeroRow).shape[0]/float(k)  
        
        self.assertEquals(precision.mean(), precisions.mean())
        
    def testRecallAtK(self): 
        m = 10 
        n = 5 
        r = 3 

        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), r, 0.5, verbose=True)

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
        
        self.assertEquals(recall.mean(), recalls.mean())
        
        #Now try random U and V 
        U = numpy.random.rand(m, 3)
        V = numpy.random.rand(m, 3)
        
        orderedItems = MCEvaluator.recommendAtk(U, V, k)
        recall, scoreInds = MCEvaluator.recallAtK(X, orderedItems, k, verbose=True)
        
        recalls = numpy.zeros(m)
        for i in range(m): 
            nonzeroRow = X.toarray()[i, :].nonzero()[0]

            recalls[i] = numpy.intersect1d(scoreInds[i, :], nonzeroRow).shape[0]/float(nonzeroRow.shape[0])    
        
        self.assertEquals(recall.mean(), recalls.mean())  
     
    def testF1Atk(self): 
        m = 10 
        n = 5 
        r = 3 
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), r, 0.5, verbose=True)

        import sppy 
        X = sppy.csarray(X)
        orderedItems = MCEvaluator.recommendAtk(U*s, V, n)

        self.assertAlmostEquals(MCEvaluator.f1AtK(X, orderedItems, n, verbose=False), 2*r/float(n)/(1+r/float(n)))
        
        
        m = 20 
        n = 50 
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), r, 0.5, verbose=True)
        k = 5

        
        
        orderedItems = MCEvaluator.recommendAtk(U*s, V, k)
        precision, scoreInds = MCEvaluator.precisionAtK(X, orderedItems, k, verbose=True)
        recall, scoreInds = MCEvaluator.recallAtK(X, orderedItems, k, verbose=True)
        f1s = numpy.zeros(m)
        
        for i in range(m): 
            f1s[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
        
        orderedItems = MCEvaluator.recommendAtk(U*s, V, n)
        f1s2, scoreInds = MCEvaluator.f1AtK(X, orderedItems, k, verbose=True)
        
        nptst.assert_array_equal(f1s, f1s2)
        
        #Test case where we get a zero precision or recall 
        orderedItems[5, :] = -1
        precision, scoreInds = MCEvaluator.precisionAtK(X, orderedItems, k, verbose=True)
        recall, scoreInds = MCEvaluator.recallAtK(X, orderedItems, k, verbose=True)
        
        f1s = numpy.zeros(m)
        
        for i in range(m): 
            if precision[i]+recall[i] != 0: 
                f1s[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
                
        f1s2, scoreInds = MCEvaluator.f1AtK(X, orderedItems, k, verbose=True)
        
        nptst.assert_array_equal(f1s, f1s2)
        
    @unittest.skip("")  
    def testLocalAUC(self): 
        m = 10 
        n = 20 
        k = 2 
        X, U, s, V,wv = SparseUtils.generateSparseBinaryMatrix((m,n), k, 0.5, verbose=True, csarray=True)
        
        Z = U.dot(V.T)

        localAuc = numpy.zeros(m)
        
        for i in range(m): 
            localAuc[i] = sklearn.metrics.roc_auc_score(numpy.ravel(X[i, :].toarray()), Z[i, :])
                    
        localAuc = localAuc.mean()
        
        u = 0.0
        localAuc2 = MCEvaluator.localAUC(X, U, V, u)

        self.assertEquals(localAuc, localAuc2)
        
        #Now try a large r 
        w = 1.0

        localAuc2 = MCEvaluator.localAUC(X, U, V, w)
        self.assertEquals(localAuc2, 0)
     
    @unittest.skip("") 
    def testLocalAucApprox(self): 
        m = 100 
        n = 200 
        k = 2 
        X, U, s, V,wv = SparseUtils.generateSparseBinaryMatrix((m, n), k, csarray=True, verbose=True)
        
        w = 1.0
        localAuc = MCEvaluator.localAUC(X, U, V, w)
        
        samples = numpy.arange(150, 200, 10)
        
        for i, sampleSize in enumerate(samples): 
            numAucSamples = sampleSize
            localAuc2 = MCEvaluator.localAUCApprox(SparseUtils.getOmegaListPtr(X), U, V, w, numAucSamples)
            self.assertAlmostEqual(localAuc2, localAuc, 1) 
            
        #Try smaller w 
        w = 0.5
        localAuc = MCEvaluator.localAUC(X, U, V, w)
        
        samples = numpy.arange(50, 200, 10)
        
        for i, sampleSize in enumerate(samples): 
            numAucSamples = sampleSize
            localAuc2 = MCEvaluator.localAUCApprox(SparseUtils.getOmegaListPtr(X), U, V, w, numAucSamples)

            self.assertAlmostEqual(localAuc2, localAuc, 1)   
       
    @unittest.skip("")    
    def testLocalAucApprox2(self): 
        m = 100 
        n = 200 
        k = 5 
        numInds = 100
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m, n), k, csarray=True, verbose=True)
        

        r = numpy.ones(m)*-10


        w = 0.5
        localAuc = MCEvaluator.localAUC(X, U, V, w)
        
        samples = numpy.arange(50, 200, 10)
        
        for i, sampleSize in enumerate(samples): 
            localAuc2 = MCEvaluator.localAUCApprox(SparseUtils.getOmegaListPtr(X), U, V, w, sampleSize)

            self.assertAlmostEqual(localAuc2, localAuc, 1)
        
        #Test more accurately 
        sampleSize = 1000
        localAuc2 = MCEvaluator.localAUCApprox(SparseUtils.getOmegaListPtr(X), U, V, w, sampleSize)
        self.assertAlmostEqual(localAuc2, localAuc, 2)
        
        #Now set a high r 
        Z = U.dot(V.T)
        localAuc = MCEvaluator.localAUCApprox(SparseUtils.getOmegaListPtr(X), U, V, w, sampleSize)  

        for i, sampleSize in enumerate(samples): 
            localAuc2 = MCEvaluator.localAUCApprox(SparseUtils.getOmegaListPtr(X), U, V, w, sampleSize)

            self.assertAlmostEqual(localAuc2, localAuc, 1)
            
        #Test more accurately 
        sampleSize = 1000
        localAuc2 = MCEvaluator.localAUCApprox(SparseUtils.getOmegaListPtr(X), U, V, w, sampleSize)
        self.assertAlmostEqual(localAuc2, localAuc, 2)       
       
    def testAverageRocCurve(self): 
        m = 50
        n = 20
        k = 8 
        u = 20.0/m
        w = 1-u
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
        
        fpr, tpr = MCEvaluator.averageRocCurve(X, U, V)
        
    @unittest.skip("") 
    def testAverageAuc(self): 
        m = 50
        n = 20
        k = 8 
        u = 20.0/m
        w = 1-u
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
        
        auc = MCEvaluator.averageAuc(X, U, V) 
        
        u = 0.0
        auc2 = MCEvaluator.localAUC(X, U, V, u)
        
        self.assertAlmostEquals(auc, auc2)        
        
if __name__ == '__main__':
    unittest.main()

