import numpy 
from sandbox.util.Util import Util 
from sandbox.util.SparseUtils import SparseUtils 
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
    def precisionAtK(X, U, V, k): 
        """
        Compute the average precision@k score for each row of the predicted matrix UV.T 
        using real values in X. X is a 0/1 sppy sparse matrix. 
        """        
        precisions = numpy.zeros(X.shape[0])
        blocksize = 500
        numBlocks = int(ceil(X.shape[0]/float(blocksize)))
        
        for j in range(numBlocks):
            endInd = min(X.shape[0], (j+1)*blocksize)
            scores = U[j*blocksize:endInd, :].dot(V.T)            
            
            for i in range(scores.shape[0]): 
                #scores = U[i, :].dot(V.T)
                #scoreInds = numpy.flipud(numpy.argsort(scores))[0:k]
                scoreInds = Util.argmaxN(scores[i, :], k)
                nonzeroRowi = X.rowInds(j*blocksize + i)
                precisions[j*blocksize + i] = numpy.intersect1d(nonzeroRowi, scoreInds).shape[0]/float(k)
        
        return precisions.mean()
      
    @staticmethod 
    def localAUC(X, U, V, u, omegaList=None, numRowInds=50): 
        """
        Compute the local AUC for the score functions UV^T relative to X with 
        quantile 1-u. 
        """
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        r = Util.computeR(U, V, 1-u, numRowInds)
        
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
    def localAUCApprox(X, U, V, u, numAucSamples, omegaList=None): 
        """
        Compute the estimated local AUC for the score functions UV^T relative to X with 
        quantile 1-u. 
        """
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        localAuc = numpy.zeros(X.shape[0]) 
        allInds = numpy.arange(X.shape[1])
        
        r = Util.computeR(U, V, 1-u, numAucSamples)
        
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
        
        