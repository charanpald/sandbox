import numpy 
import logging
from sandbox.util.Util import Util 
from sandbox.util.SparseUtils import SparseUtils 
from sandbox.util.SparseUtilsCython import SparseUtilsCython 
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
from math import ceil 

class MCEvaluator(object):
    """
    A class to evaluate machine learning performance for the matrix completion
    problem.
    """
    def __init__(self):
        pass
    
    @staticmethod 
    def meanSqError(testX, predX): 
        """
        Find the mean squared error between two sparse matrices testX and predX. 
        Note that the matrices must have nonzero elements in the same places. 
        """
        #Note that some predictions might be zero 
        assert numpy.in1d(predX.nonzero()[0], testX.nonzero()[0]).all() 
        assert numpy.in1d(predX.nonzero()[1], testX.nonzero()[1]).all() 
        
        diff = testX - predX     
        error = numpy.sum(diff.data**2)/testX.data.shape[0]
        return error

        
    @staticmethod 
    def rootMeanSqError(testX, predX): 
        """
        Find the root mean squared error between two sparse matrices testX and predX. 
        """
        
        return numpy.sqrt(MCEvaluator.meanSqError(testX, predX)) 
        
    @staticmethod 
    def meanAbsError(testX, predX): 
        """
        Find the mean absolute error between two sparse matrices testX and predX. 
        Note that the matrices must have nonzero elements in the same places. 
        """
        #Note that some predictions might be zero 
        assert numpy.in1d(predX.nonzero()[0], testX.nonzero()[0]).all() 
        assert numpy.in1d(predX.nonzero()[1], testX.nonzero()[1]).all() 
        
        diff = testX - predX     
        error = numpy.abs(diff.data).sum()/testX.data.shape[0]
        return error
        
    @staticmethod 
    def precisionAtK(positiveArray, orderedItems, k, verbose=False): 
        """
        Compute the average precision@k score for each row of the predicted matrix UV.T 
        using real values in positiveArray. positiveArray is a tuple (indPtr, colInds)
        
        :param orderedItems: The ordered items for each user (users are rows, items are cols)       
        
        :param verbose: If true return precision and first k recommendation for each row, otherwise just precisions
        """
        if type(positiveArray) != tuple: 
            positiveArray = SparseUtils.getOmegaListPtr(positiveArray)
        
        orderedItems = orderedItems[:, 0:k]
        indPtr, colInds = positiveArray
        precisions = MCEvaluatorCython.precisionAtk(indPtr, colInds, orderedItems)
        
        if verbose: 
            return precisions, orderedItems
        else: 
            return precisions.mean()

    @staticmethod 
    def recallAtK(positiveArray, orderedItems, k, verbose=False): 
        """
        Compute the average recall@k score for each row of the predicted matrix UV.T 
        using real values in positiveArray. positiveArray is a tuple (indPtr, colInds)
        
        :param orderedItems: The ordered items for each user (users are rows, items are cols)  
        
        :param verbose: If true return recall and first k recommendation for each row, otherwise just precisions
        """
        if type(positiveArray) != tuple: 
            positiveArray = SparseUtils.getOmegaListPtr(positiveArray)        
        
        orderedItems = orderedItems[:, 0:k]
        indPtr, colInds = positiveArray
        recalls = MCEvaluatorCython.recallAtk(indPtr, colInds, orderedItems)
        
        if verbose: 
            return recalls, orderedItems
        else: 
            return recalls.mean()
            
    @staticmethod       
    def f1AtK(positiveArray, orderedItems, k, verbose=False): 
        """
        Return the F1@k measure for each row of the predicted matrix UV.T 
        using real values in positiveArray. positiveArray is a tuple (indPtr, colInds)
        
        :param orderedItems: The ordered items for each user (users are rows, items are cols)  
        
        :param verbose: If true return recall and first k recommendation for each row, otherwise just precisions
        """
        if type(positiveArray) != tuple: 
            positiveArray = SparseUtils.getOmegaListPtr(positiveArray)        
        
        orderedItems = orderedItems[:, 0:k]
        indPtr, colInds = positiveArray
        
        precisions = MCEvaluatorCython.precisionAtk(indPtr, colInds, orderedItems)
        recalls = MCEvaluatorCython.recallAtk(indPtr, colInds, orderedItems)
        
        denominator = precisions+recalls
        denominator += denominator == 0      
        
        f1s = 2*precisions*recalls/denominator
        
        if verbose: 
            return f1s, orderedItems
        else: 
            return f1s.mean()

    @staticmethod 
    def recommendAtk(U, V, k, blockSize=1000, omegaList=None): 
        """
        Compute the matrix Z = U V^T and then find the k largest indices for each row. 
        """
        blocksize = 1000
        numBlocks = int(ceil(U.shape[0]/float(blocksize)))
        orderedItems = numpy.zeros((U.shape[0], k), numpy.int32)

        for j in range(numBlocks):
            logging.debug("Block " + str(j) + " of " + str(numBlocks))
            endInd = min(U.shape[0], (j+1)*blocksize)
            scores = U[j*blocksize:endInd, :].dot(V.T)     
            orderedItems[j*blocksize:endInd, :] = Util.argmaxN(scores, k)
            #orderedItems[j*blocksize:endInd, :] = Util.argmaxN2d(scores, k)
            
            #Now delete items in omegaList if given 
            if omegaList != None: 
                for i in range(j*blocksize, endInd):
                    
                    nonTrainItems = orderedItems[i, :][numpy.logical_not(numpy.in1d(orderedItems[i, :], omegaList[i]))]
                    orderedItems[i, 0:nonTrainItems.shape[0]] = nonTrainItems
                    orderedItems[i, nonTrainItems.shape[0]:] = -1
            
        return orderedItems 
        
    @staticmethod 
    def localAUC(positiveArray, U, V, w, numRowInds=None): 
        """
        Compute the local AUC for the score functions UV^T relative to X with 
        quantile w. 
        """
        if numRowInds == None: 
            numRowInds = V.shape[0]
            
        if type(positiveArray) != tuple: 
            positiveArray = SparseUtils.getOmegaListPtr(positiveArray)  
        
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        r = SparseUtilsCython.computeR(U, V, w, numRowInds)
        
        localAuc = numpy.zeros(U.shape[0]) 
        allInds = numpy.arange(V.shape[0])
        indPtr, colInds = positiveArray
        
        for i in range(U.shape[0]): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0                
                
                for p in omegai: 
                    for q in omegaBari: 
                        if Z[i, p] > Z[i, q] and Z[i, p] > r[i]: 
                            partialAuc += 1 
                            
                localAuc[i] = partialAuc/float(omegai.shape[0] * omegaBari.shape[0])
        
        localAuc = localAuc.mean()        
        
        return localAuc

    @staticmethod
    def localAUCApprox(positiveArray, U, V, w, numAucSamples=50, r=None, allArray=None): 
        """
        Compute the estimated local AUC for the score functions UV^T relative to X with 
        quantile w. The AUC is computed using positiveArray which is a tuple (indPtr, colInds)
        assuming allArray is None. If allArray is not None then positive items are chosen 
        from positiveArray and negative ones are chosen to complement allArray.
        """
        
        if type(positiveArray) != tuple: 
            positiveArray = SparseUtils.getOmegaListPtr(positiveArray)          
        
        indPtr, colInds = positiveArray
        U = numpy.ascontiguousarray(U)
        V = numpy.ascontiguousarray(V)        
        
        if r == None: 
            r = SparseUtilsCython.computeR(U, V, w, numAucSamples)
        
        if allArray == None: 
            return MCEvaluatorCython.localAUCApprox(indPtr, colInds, indPtr, colInds, U, V, numAucSamples, r)
        else:
            allIndPtr, allColInd = allArray
            return MCEvaluatorCython.localAUCApprox(indPtr, colInds, allIndPtr, allColInd, U, V, numAucSamples, r)
            


    @staticmethod
    def localAUCApprox2(X, U, V, w, numAucSamples=50, omegaList=None): 
        """
        Compute the estimated local AUC for the score functions UV^T relative to X with 
        quantile w. 
        """
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        localAuc = numpy.zeros(X.shape[0]) 
        allInds = numpy.arange(X.shape[1])
        
        U = numpy.ascontiguousarray(U)
        V = numpy.ascontiguousarray(V)
        
        r = SparseUtilsCython.computeR(U, V, w, numAucSamples)
        
        if omegaList==None: 
            omegaList = SparseUtils.getOmegaList(X)

        for i in range(X.shape[0]): 
            omegai = omegaList[i]
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0 

                for j in range(numAucSamples):
                    ind = numpy.random.randint(omegai.shape[0]*omegaBari.shape[0])
                    p = omegai[int(ind/omegaBari.shape[0])] 
                    q = omegaBari[ind % omegaBari.shape[0]]   
                    
                    if Z[i, p] > Z[i, q] and Z[i, p] > r[i]: 
                        partialAuc += 1 
                            
                localAuc[i] = partialAuc/float(numAucSamples)
          
        localAuc = localAuc.mean()        
        
        return localAuc        
        
    @staticmethod 
    def averageRocCurve(X, U, V): 
        """
        Compute the average ROC curve for the rows of X given preductions based 
        on U V^T. 
        """
        import sklearn.metrics 
        (m, n) = X.shape 
        tprs = numpy.zeros(n)
        fprs = numpy.zeros(n)
        
        for i in range(m): 
            trueXi = X[i, :].toarray().ravel()
            predXi = U[i, :].T.dot(V.T)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(trueXi, predXi)

            #Sometimes the fpr and trp are not length m (not sure why) so make them fit 
            fprs += fpr[0:n] 
            tprs += tpr[0:n] 
            
        fprs /= m
        tprs /= m 
        
        return fprs, tprs 
        
    @staticmethod 
    def averageAuc(X, U, V): 
        """
        Compute the average ROC curve for the rows of X given preductions based 
        on U V^T. 
        """
        import sklearn.metrics 
        (m, n) = X.shape 
        aucs = numpy.zeros(m)
        
        for i in range(m): 
            trueXi = X[i, :].toarray().ravel()
            predXi = U[i, :].T.dot(V.T)
            aucs[i] = sklearn.metrics.roc_auc_score(trueXi, predXi)
        
        return aucs.mean()         
        