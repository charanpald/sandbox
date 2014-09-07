
from sandbox.util.Parameter import Parameter
from sandbox.util.SparseUtils import SparseUtils
import numpy
import array 
import scipy.sparse


class Sampling(object):
    """
    An class to sample a set of examples in different ways. 
    """
    def __init__(self):
        pass

    @staticmethod
    def crossValidation(folds, numExamples):
        """
        Returns a list of tuples (trainIndices, testIndices) using k-fold cross
        validation. The dataset is split into approximately folds contiguous 
        subsamples.  

        :param folds: The number of cross validation folds.
        :type folds: :class:`int`

        :param numExamples: The number of examples.
        :type numExamples: :class:`int`
        """
        Parameter.checkInt(folds, 1, numExamples)
        Parameter.checkInt(numExamples, 2, float('inf'))

        foldSize = float(numExamples)/folds
        indexList = []

        for i in range(0, folds):
            testIndices = numpy.arange(int(foldSize*i), int(foldSize*(i+1)))
            trainIndices = numpy.setdiff1d(numpy.arange(0, numExamples), numpy.array(testIndices))
            indexList.append((trainIndices, testIndices))

        return indexList 

    @staticmethod
    def randCrossValidation(folds, numExamples, dtype=numpy.int32):
        """
        Returns a list of tuples (trainIndices, testIndices) using k-fold cross
        validation. In this case we randomise the indices and then split into 
        folds. 

        :param folds: The number of cross validation folds.
        :type folds: :class:`int`

        :param numExamples: The number of examples.
        :type numExamples: :class:`int`
        """
        Parameter.checkInt(folds, 1, numExamples)
        Parameter.checkInt(numExamples, 2, float('inf'))

        foldSize = float(numExamples)/folds
        indexList = []

        inds = numpy.array(numpy.random.permutation(numExamples), dtype)

        for i in range(0, folds):
            testIndices = inds[int(foldSize*i): int(foldSize*(i+1))]
            trainIndices = numpy.setdiff1d(numpy.arange(0, numExamples), testIndices)
            indexList.append((trainIndices, testIndices))

        return indexList 

    @staticmethod
    def bootstrap(repetitions, numExamples):
        """
        Perform 0.632 bootstrap in whcih we take a sample with replacement from
        the dataset of size numExamples. The examples not present in the training
        set are used to form the test set. Returns a list of tuples of the form
        (trainIndices, testIndices).

        :param repetitions: The number of repetitions of bootstrap to perform.
        :type repetitions: :class:`int`

        :param numExamples: The number of examples.
        :type numExamples: :class:`int`

        """
        Parameter.checkInt(numExamples, 2, float('inf'))
        Parameter.checkInt(repetitions, 1, float('inf'))

        inds = []
        for i in range(repetitions):
            trainInds = numpy.random.randint(numExamples, size=numExamples)
            testInds = numpy.setdiff1d(numpy.arange(numExamples), numpy.unique(trainInds))
            inds.append((trainInds, testInds))

        return inds

    @staticmethod
    def bootstrap2(repetitions, numExamples):
        """
        Perform 0.632 bootstrap in whcih we take a sample with replacement from
        the dataset of size numExamples. The examples not present in the training
        set are used to form the test set. We oversample the test set to include
        0.368 of the examples from the training set. Returns a list of tuples of the form
        (trainIndices, testIndices).

        :param repetitions: The number of repetitions of bootstrap to perform.
        :type repetitions: :class:`int`

        :param numExamples: The number of examples.
        :type numExamples: :class:`int`

        """
        Parameter.checkInt(numExamples, 2, float('inf'))
        Parameter.checkInt(repetitions, 1, float('inf'))

        inds = []
        for i in range(repetitions):
            trainInds = numpy.random.randint(numExamples, size=numExamples)
            testInds = numpy.setdiff1d(numpy.arange(numExamples), numpy.unique(trainInds))
            #testInds = numpy.r_[testInds, trainInds[0:(numExamples*0.368)]]

            inds.append((trainInds, testInds))

        return inds

    @staticmethod 
    def shuffleSplit(repetitions, numExamples, trainProportion=None):
        """
        Random permutation cross-validation iterator. The training set is sampled
        without replacement and of size (repetitions-1)/repetitions of the examples,
        and the test set represents the remaining examples. Each repetition is
        sampled independently.

        :param repetitions: The number of repetitions to perform.
        :type repetitions: :class:`int`

        :param numExamples: The number of examples.
        :type numExamples: :class:`int`

        :param trainProp: The size of the training set relative to numExamples, between 0 and 1 or None to use (repetitions-1)/repetitions
        :type trainProp: :class:`int`
        """
        Parameter.checkInt(numExamples, 2, float('inf'))
        Parameter.checkInt(repetitions, 1, float('inf'))
        if trainProportion != None:
            Parameter.checkFloat(trainProportion, 0.0, 1.0)

        if trainProportion == None:
            trainSize = int((repetitions-1)*numExamples/repetitions)
        else:
            trainSize = int(trainProportion*numExamples)

        idx = [] 
        for i in range(repetitions):
            inds = numpy.random.permutation(numExamples)
            trainInds = inds[0:trainSize]
            testInds = inds[trainSize:]
            idx.append((trainInds, testInds))
        return idx 
        
    @staticmethod
    def repCrossValidation(folds, numExamples, repetitions, seed=21):
        """
        Returns a list of tuples (trainIndices, testIndices) using k-fold cross
        validation repeated m times. 

        :param folds: The number of cross validation folds.
        :type folds: :class:`int`

        :param numExamples: The number of examples.
        :type numExamples: :class:`int`
        
        :param repetitions: The number of repetitions.
        :type repetitions: :class:`int`
        """
        Parameter.checkInt(folds, 1, numExamples)
        Parameter.checkInt(numExamples, 2, float('inf'))
        Parameter.checkInt(repetitions, 1, float('inf'))

        foldSize = float(numExamples)/folds
        indexList = []
        numpy.random.seed(seed)
        
        for j in range(repetitions): 
            permInds = numpy.random.permutation(numExamples)
            
            for i in range(folds):
                testIndices = numpy.arange(int(foldSize*i), int(foldSize*(i+1)))
                trainIndices = numpy.setdiff1d(numpy.arange(0, numExamples), numpy.array(testIndices))
                indexList.append((permInds[trainIndices], permInds[testIndices]))

        return indexList 
        
    @staticmethod 
    def shuffleSplitRows(X, k, testSize, numRows=None, csarray=True, rowMajor=True, colProbs=None): 
        """
        Take a sparse binary matrix and create k number of train-test splits 
        in which the test split contains at most testSize elements and the train 
        split contains the remaining elements from X for each row. The splits are 
        computed randomly. Returns sppy.csarray objects by default. 
        
        :param colProbs: This is the probability of choosing the corresponding column/item. If None, we assume uniform probabilities. 
        """
        if csarray: 
            mattype = "csarray"
        else: 
            mattype = "scipy" 
            
        if rowMajor: 
            storagetype = "row" 
        else: 
            storagetype = "col"
            
        if numRows == None: 
            numRows = X.shape[0]
            outputRows = False
        else: 
            outputRows = True
        
        trainTestXList = []
        omegaList = SparseUtils.getOmegaList(X)
        m, n = X.shape
        
        for i in range(k):
            trainInd = 0 
            testInd = 0            
            
            trainRowInds = numpy.zeros(X.nnz, numpy.int32)
            trainColInds = numpy.zeros(X.nnz, numpy.int32)
            
            testRowInds = numpy.zeros(X.shape[0]*testSize, numpy.int32)
            testColInds = numpy.zeros(X.shape[0]*testSize, numpy.int32)

            rowSample = numpy.sort(numpy.random.choice(m, numRows, replace=False))

            for j in range(m):
                
                if j in rowSample: 
                    if colProbs == None: 
                        inds = numpy.random.permutation(omegaList[j].shape[0])
                    else: 
                        probs = colProbs[omegaList[j]]
                        probs /= probs.sum() 
                        inds = numpy.random.choice(omegaList[j].shape[0], omegaList[j].shape[0], p=probs, replace=False)
                    trainInds = inds[testSize:]
                    testInds = inds[0:testSize]
                else: 
                    trainInds = numpy.arange(omegaList[j].shape[0]) 
                    testInds = numpy.array([], numpy.int)
                    
                trainRowInds[trainInd:trainInd+trainInds.shape[0]] = numpy.ones(trainInds.shape[0], dtype=numpy.uint)*j
                trainColInds[trainInd:trainInd+trainInds.shape[0]] = omegaList[j][trainInds]
                trainInd += trainInds.shape[0]
                
                testRowInds[testInd:testInd+testInds.shape[0]] = numpy.ones(testInds.shape[0], dtype=numpy.uint)*j
                testColInds[testInd:testInd+testInds.shape[0]] = omegaList[j][testInds]
                testInd += testInds.shape[0]
                
            trainRowInds = trainRowInds[0:trainInd]   
            trainColInds = trainColInds[0:trainInd] 
      
            testRowInds = testRowInds[0:testInd]   
            testColInds = testColInds[0:testInd]
            
            trainX = SparseUtils.sparseMatrix(numpy.ones(trainRowInds.shape[0], numpy.int), trainRowInds, trainColInds, X.shape, mattype, storagetype)
            testX = SparseUtils.sparseMatrix(numpy.ones(testRowInds.shape[0], numpy.int), testRowInds, testColInds, X.shape, mattype, storagetype)

            if not outputRows: 
                trainTestXList.append((trainX, testX))
            else: 
                trainTestXList.append((trainX, testX, rowSample))
        
        return trainTestXList 
        
    @staticmethod 
    def sampleUsers(X, k): 
        """
        Let's pick a sample of users from X such that each item has at least 2 
        users. Pick at most k users. 
        """
        m, n = X.shape        
        
        if X.shape[0] <= k: 
            return X 
        else: 
            itemInds = numpy.random.permutation(n)
            userInds = numpy.array([], numpy.int32)
            i = 0
            
            while len(userInds) < k and i < n: 
                userInds = numpy.union1d(userInds, numpy.nonzero(X[:, itemInds[i]].sum(1))[0])

                i += 1
                
            userInds = numpy.random.choice(userInds, min(k, userInds.shape[0]), replace=False)
            userInds = numpy.sort(userInds)
            
            return X[userInds, :]
        
