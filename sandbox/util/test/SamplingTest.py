
import unittest
import numpy 
import numpy.testing as nptst 
from sandbox.util.Sampling import Sampling 
from sandbox.util.SparseUtils import SparseUtils 

class  SamplingTest(unittest.TestCase):
    def testCrossValidation(self):
        numExamples = 10
        folds = 2

        indices = Sampling.crossValidation(folds, numExamples)

        self.assertEquals((list(indices[0][0]), list(indices[0][1])), ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4]))
        self.assertEquals((list(indices[1][0]), list(indices[1][1])), ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]))

        indices = Sampling.crossValidation(3, numExamples)

        self.assertEquals((list(indices[0][0]), list(indices[0][1])), ([3, 4, 5, 6, 7, 8, 9], [0, 1, 2]))
        self.assertEquals((list(indices[1][0]), list(indices[1][1])), ([0, 1, 2, 6, 7, 8, 9], [3, 4, 5]))
        self.assertEquals((list(indices[2][0]), list(indices[2][1])), ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9]))

        indices = Sampling.crossValidation(4, numExamples)

        self.assertEquals((list(indices[0][0]), list(indices[0][1])), ([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]))
        self.assertEquals((list(indices[1][0]), list(indices[1][1])), ([0, 1, 5, 6, 7, 8, 9], [2, 3, 4]))
        self.assertEquals((list(indices[2][0]), list(indices[2][1])), ([0, 1, 2, 3, 4, 7, 8, 9], [5, 6]))
        self.assertEquals((list(indices[3][0]), list(indices[3][1])), ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9]))

        indices = Sampling.crossValidation(numExamples, numExamples)
        self.assertEquals((list(indices[0][0]), list(indices[0][1])), ([1, 2, 3, 4, 5, 6, 7, 8, 9], [0]))
        self.assertEquals((list(indices[1][0]), list(indices[1][1])), ([0, 2, 3, 4, 5, 6, 7, 8, 9], [1]))
        self.assertEquals((list(indices[2][0]), list(indices[2][1])), ([0, 1, 3, 4, 5, 6, 7, 8, 9], [2]))
        self.assertEquals((list(indices[3][0]), list(indices[3][1])), ([0, 1, 2, 4, 5, 6, 7, 8, 9], [3]))
        self.assertEquals((list(indices[4][0]), list(indices[4][1])), ([0, 1, 2, 3, 5, 6, 7, 8, 9], [4]))

        self.assertRaises(ValueError, Sampling.crossValidation, numExamples+1, numExamples)
        self.assertRaises(ValueError, Sampling.crossValidation, 0, numExamples)
        self.assertRaises(ValueError, Sampling.crossValidation, -1, numExamples)
        self.assertRaises(ValueError, Sampling.crossValidation, folds, 1)

    def testBootstrap(self):
        numExamples = 10
        folds = 2

        indices = Sampling.bootstrap(folds, numExamples)

        for i in range(folds): 
            self.assertEquals(indices[i][0].shape[0], numExamples)
            self.assertTrue(indices[i][1].shape[0] < numExamples)
            self.assertTrue((numpy.union1d(indices[0][0], indices[0][1]) == numpy.arange(numExamples)).all())

    def testBootstrap2(self):
        numExamples = 10
        folds = 2

        indices = Sampling.bootstrap2(folds, numExamples)

        for i in range(folds):
            self.assertEquals(indices[i][0].shape[0], numExamples)
            self.assertTrue(indices[i][1].shape[0] < numExamples)
            self.assertTrue((numpy.union1d(indices[0][0], indices[0][1]) == numpy.arange(numExamples)).all())


    def testShuffleSplit(self):
        numExamples = 10
        folds = 5

        indices = Sampling.shuffleSplit(folds, numExamples)
        
        for i in range(folds):
            self.assertTrue((numpy.union1d(indices[i][0], indices[i][1]) == numpy.arange(numExamples)).all())
        
        indices = Sampling.shuffleSplit(folds, numExamples, 0.5)
        trainSize = numExamples*0.5

        for i in range(folds):
            self.assertTrue((numpy.union1d(indices[i][0], indices[i][1]) == numpy.arange(numExamples)).all())
            self.assertTrue(indices[i][0].shape[0] == trainSize)

        indices = Sampling.shuffleSplit(folds, numExamples, 0.55)
    
    def testRepCrossValidation(self): 
        numExamples = 10
        folds = 3
        repetitions = 1

        indices = Sampling.repCrossValidation(folds, numExamples, repetitions)
        
        for i in range(folds):
            self.assertTrue((numpy.union1d(indices[i][0], indices[i][1]) == numpy.arange(numExamples)).all())
        
        repetitions = 2
        indices = Sampling.repCrossValidation(folds, numExamples, repetitions)
        
        for i in range(folds):
            self.assertTrue((numpy.union1d(indices[i][0], indices[i][1]) == numpy.arange(numExamples)).all())

    def testRandCrossValidation(self): 
        numExamples = 10
        folds = 3
        
        indices = Sampling.randCrossValidation(folds, numExamples)
    
        
        for i in range(folds):
            self.assertTrue((numpy.union1d(indices[i][0], indices[i][1]) == numpy.arange(numExamples)).all())
        

    def testShuffleSplitRows(self): 
        m = 10
        n = 16
        k = 5 
        u = 0.5
        w = 1-u
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
        
        #print(X.toarray())
        
        k2 = 5 
        testSize = 2
        trainTestXs = Sampling.shuffleSplitRows(X, k2, testSize, rowMajor=True)
        
        for i in range(k2): 
            trainX = trainTestXs[i][0]
            testX = trainTestXs[i][1]
                        
            self.assertEquals(trainX.storagetype, "row")
            self.assertEquals(testX.storagetype, "row")
            nptst.assert_array_almost_equal(X.toarray(), (trainX+testX).toarray())
            nptst.assert_array_equal(testX.sum(1), testSize*numpy.ones(m))
            self.assertEquals(X.nnz, trainX.nnz + testX.nnz)
        
        trainTestXs = Sampling.shuffleSplitRows(X, k2, testSize, rowMajor=False)
        
        for i in range(k2): 
            trainX = trainTestXs[i][0]
            testX = trainTestXs[i][1]
                       
            self.assertEquals(trainX.storagetype, "col")
            self.assertEquals(testX.storagetype, "col")                       
            nptst.assert_array_almost_equal(X.toarray(), (trainX+testX).toarray())
            nptst.assert_array_equal(testX.sum(1), testSize*numpy.ones(m))
            self.assertEquals(X.nnz, trainX.nnz + testX.nnz)        
        
        trainTestXs = Sampling.shuffleSplitRows(X, k2, testSize, csarray=False)
        for i in range(k2): 
            trainX = trainTestXs[i][0]
            testX = trainTestXs[i][1]
                        
            nptst.assert_array_almost_equal(X.toarray(), (trainX+testX).toarray())
            
            nptst.assert_array_equal(numpy.ravel(testX.sum(1)), testSize*numpy.ones(m))
            self.assertEquals(X.nnz, trainX.nnz + testX.nnz)

        testSize = 0
        trainTestXs = Sampling.shuffleSplitRows(X, k2, testSize)
        
        for i in range(k2): 
            trainX = trainTestXs[i][0]
            testX = trainTestXs[i][1]
                        
            nptst.assert_array_almost_equal(X.toarray(), (trainX+testX).toarray())
            nptst.assert_array_equal(testX.sum(1), testSize*numpy.ones(m))
            self.assertEquals(X.nnz, trainX.nnz + testX.nnz)
            self.assertEquals(testX.nnz, 0)
            
        #Test sampling a subset of the rows 
        testSize = 2
        numRows = 5
        trainTestXs = Sampling.shuffleSplitRows(X, k2, testSize, numRows=numRows, rowMajor=False)

        for i in range(k2): 
            trainX = trainTestXs[i][0]
            testX = trainTestXs[i][1]
            
            nptst.assert_array_almost_equal(X.toarray(), (trainX+testX).toarray())
            self.assertEquals(numpy.nonzero(testX.sum(1))[0].shape[0], numRows)
            self.assertEquals(X.nnz, trainX.nnz + testX.nnz)
            self.assertEquals(testX.nnz, testSize*numRows)
            
        #Make sure column probabilities are correct 
        w = 0.0            
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)            
            
        testSize = 5
        k2 = 500
        colProbs = numpy.arange(0, n, dtype=numpy.float)+1
        colProbs /= colProbs.sum() 
        trainTestXs = Sampling.shuffleSplitRows(X, k2, testSize, colProbs=colProbs)
        
        colProbs2 = numpy.zeros(n)        
        
        for i in range(k2): 
            trainX = trainTestXs[i][0]
            testX = trainTestXs[i][1]
            
            colProbs2 += testX.sum(0)
        
        colProbs2 /= colProbs2.sum() 
        nptst.assert_array_almost_equal(colProbs, colProbs2, 2)
        
        #Now test when probabilities are uniform 
        colProbs = numpy.ones(n)/float(n)        
        trainTestXs = Sampling.shuffleSplitRows(X, k2, testSize, colProbs=colProbs)
        
        colProbs = None
        trainTestXs2 = Sampling.shuffleSplitRows(X, k2, testSize, colProbs=colProbs)
        
        colProbs2 = numpy.zeros(n)       
        colProbs3 = numpy.zeros(n) 
        
        for i in range(k2): 
            trainX = trainTestXs[i][0]
            testX = trainTestXs[i][1]
            colProbs2 += testX.sum(0)
            
            trainX = trainTestXs2[i][0]
            testX = trainTestXs2[i][1]
            colProbs3 += testX.sum(0)
        
        colProbs2 /= colProbs2.sum() 
        colProbs3 /= colProbs3.sum()
        nptst.assert_array_almost_equal(colProbs2, colProbs3, 2)
        
        #Test when numRows=m
        numpy.random.seed(21)
        trainTestXs = Sampling.shuffleSplitRows(X, k2, testSize, numRows=m)
        numpy.random.seed(21)
        trainTestXs2 = Sampling.shuffleSplitRows(X, k2, testSize)

        nptst.assert_array_equal(trainTestXs[0][0].toarray(), trainTestXs2[0][0].toarray())
        nptst.assert_array_equal(trainTestXs[0][1].toarray(), trainTestXs2[0][1].toarray())

    def testSampleUsers(self): 
        m = 10
        n = 15
        r = 5 
        u = 0.3
        w = 1-u
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), r, w, csarray=True, verbose=True, indsPerRow=200)

        k = 50
        X2 = Sampling.sampleUsers(X, k)

        nptst.assert_array_equal(X.toarray(), X2.toarray())
        
        numRuns = 50
        for i in range(numRuns): 
            m = numpy.random.randint(10, 100)
            n = numpy.random.randint(10, 100)
            k = numpy.random.randint(10, 100)

            X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), r, w, csarray=True, verbose=True, indsPerRow=200)

            X2 = Sampling.sampleUsers(X, k)
            
            self.assertEquals(X2.shape[0], min(k, m))
            self.assertTrue((X.dot(X.T)!=numpy.zeros((m, m)).all()))
        

if __name__ == '__main__':
    unittest.main()

