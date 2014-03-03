import numpy 
from sandbox.util.Util import Util 
from sandbox.util.SparseUtils import SparseUtils 
from sandbox.util.SparseUtilsCython import SparseUtilsCython 
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
    def precisionAtK(X, U, V, k, scoreInds=None, omegaList=None, verbose=False): 
        """
        Compute the average precision@k score for each row of the predicted matrix UV.T 
        using real values in X. X is a 0/1 sppy sparse matrix.
        
        :param verbose: If true return precision and first k recommendation for each row, otherwise just precisions
        """
        if scoreInds == None: 
            scoreInds = MCEvaluator.recommendAtk(U, V, k)
        if omegaList == None: 
            omegaList = SparseUtils.getOmegaList(X)
        
        precisions = numpy.zeros(X.shape[0])
        precisions = SparseUtilsCython.precisionAtk(omegaList, scoreInds)

        if verbose: 
            return precisions.mean(), scoreInds
        else: 
            return precisions.mean()

    @staticmethod 
    def recommendAtk(U, V, k, blockSize=1000): 
        """
        Compute the matrix Z = U V^T and then find the k largest indices for each row. 
        """
        blocksize = 1000
        numBlocks = int(ceil(U.shape[0]/float(blocksize)))
        scoreInds = numpy.zeros((U.shape[0], k), numpy.int32)

        for j in range(numBlocks):
            endInd = min(U.shape[0], (j+1)*blocksize)
            scores = U[j*blocksize:endInd, :].dot(V.T)     
            scoreInds[j*blocksize:endInd, :] = Util.argmaxN(scores, k)
            #scoreInds[j*blocksize:endInd, :] = Util.argmaxN2d(scores, k)
            
        return scoreInds 
        
    @staticmethod 
    def localAUC(X, U, V, u, omegaList=None, numRowInds=None): 
        """
        Compute the local AUC for the score functions UV^T relative to X with 
        quantile 1-u. 
        """
        if numRowInds == None: 
            numRowInds = V.shape[0]
        
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        r = SparseUtilsCython.computeR(U, V, 1-u, numRowInds)
        
        if omegaList==None: 
            omegaList = SparseUtils.getOmegaList(X)
        
        localAuc = numpy.zeros(X.shape[0]) 
        allInds = numpy.arange(X.shape[1])
        
        for i in range(X.shape[0]): 
            omegai = omegaList[i]
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
    def localAUCApprox(X, U, V, u, numAucSamples=50, omegaList=None): 
        """
        Compute the estimated local AUC for the score functions UV^T relative to X with 
        quantile 1-u. 
        """
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        localAuc = numpy.zeros(X.shape[0]) 
        allInds = numpy.arange(X.shape[1])
        
        U = numpy.ascontiguousarray(U)
        V = numpy.ascontiguousarray(V)
        
        r = SparseUtilsCython.computeR(U, V, 1-u, numAucSamples)
        
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
        
        