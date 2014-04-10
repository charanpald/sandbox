
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
        X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
        
        #print(X.toarray())
        
        k2 = 5 
        testSize = 2
        trainTestXs = Sampling.shuffleSplitRows(X, k2, testSize)
        
        for i in range(k2): 
            trainX = trainTestXs[i][0]
            testX = trainTestXs[i][1]
                        
            nptst.assert_array_almost_equal(X.toarray(), (trainX+testX).toarray())
            
            nptst.assert_array_equal(testX.sum(1), testSize*numpy.ones(m))

if __name__ == '__main__':
    unittest.main()

