
import numpy 
import logging
import multiprocessing 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.util.MCEvaluator import  MCEvaluator
from sandbox.util.Sampling import Sampling 
from sandbox.util.Util import Util 
from mrec.mf.warp import WARPMFRecommender

def localAucsLmbdas(args): 
    trainX, testX, testOmegaList, learner  = args 
    
    (m, n) = trainX.shape
                        
    localAucs = numpy.zeros(learner.lmbdas.shape[0])

    for j, lmbda in enumerate(learner.lmbdas): 
        learner.lmbda = lmbda 
        
        U, V = learner.learnModel(trainX)
        
        r = SparseUtilsCython.computeR(U, V, 1-learner.u, learner.numAucSamples)
        localAucs[j] = MCEvaluator.localAUCApprox(testX, U, V, testOmegaList, learner.numAucSamples, r) 
        logging.debug("Local AUC: " + str(localAucs[j]) + " with k = " + str(learner.k) + " and lmbda= " + str(learner.lmbda))
        
    return localAucs

class WarpMf(object): 
    """
    A wrapper for the mrec class WARPMFRecommender. 
    """
    
    def __init__(self, k, alpha=0.1, lmbda=0.015, batchSize=10, maxTrials=15, u=0.1): 
        """
        :param k: The dimensionality 
        
        :param alpha: The learning rate 
        
        :param lmbda: Regualarisation constant 
        """
        self.k = k 
        self.alpha = alpha 
        self.lmbda = lmbda 
        self.batchSize = batchSize
        self.maxTrials = maxTrials 
        
        self.ks = numpy.array([10, 20, 50, 100])
        self.lmbdas = numpy.flipud(numpy.logspace(-3, -1, 11)*2) 
        
        self.folds = 3
        self.u = u
        self.numAucSamples = 100
        self.numProcesses = multiprocessing.cpu_count()
        self.chunkSize = 1
        
    def learnModel(self, X): 
        
        learner = WARPMFRecommender(self.k, self.alpha, self.lmbda, self.batchSize, self.maxTrials)
        
        learner.fit(X)
        
        return learner.U, learner.V 
        
    def modelSelect(self, X): 
        """
        Perform model selection on X and return the best parameters. 
        """
        m, n = X.shape
        cvInds = Sampling.randCrossValidation(self.folds, X.nnz)
        localAucs = numpy.zeros((self.ks.shape[0], self.lmbdas.shape[0], len(cvInds)))
        
        logging.debug("Performing model selection")
        paramList = []        
        
        for icv, (trainInds, testInds) in enumerate(cvInds):
            Util.printIteration(icv, 1, self.folds, "Fold: ")

            trainX = SparseUtils.submatrix(X, trainInds)
            testX = SparseUtils.submatrix(X, testInds)
            
            testOmegaList = SparseUtils.getOmegaList(testX)
            
            for i, k in enumerate(self.ks): 
                maxLocalAuc = self.copy()
                maxLocalAuc.k = k
                paramList.append((trainX, testX, testOmegaList, maxLocalAuc))
                    
        pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
        resultsIterator = pool.imap(localAucsLmbdas, paramList, self.chunkSize)
        #import itertools
        #resultsIterator = itertools.imap(localAucsLmbdas, paramList)
        
        for icv, (trainInds, testInds) in enumerate(cvInds):        
            for i, k in enumerate(self.ks): 
                tempAucs = resultsIterator.next()
                localAucs[i, :, icv] = tempAucs
        
        pool.terminate()
        
        meanLocalAucs = numpy.mean(localAucs, 2)
        stdLocalAucs = numpy.std(localAucs, 2)
        
        logging.debug(meanLocalAucs)
        
        k = self.ks[numpy.unravel_index(numpy.argmax(meanLocalAucs), meanLocalAucs.shape)[0]]
        lmbda = self.lmbdas[numpy.unravel_index(numpy.argmax(meanLocalAucs), meanLocalAucs.shape)[1]]
        
        logging.debug("Model parameters: k=" + str(k) + " lmbda=" + str(lmbda))
        
        self.k = k 
        self.lmbda = lmbda 
        
        return meanLocalAucs, stdLocalAucs
    
    def copy(self): 
        learner = WarpMf(self.k, self.alpha, self.lmbda, self.batchSize, self.maxTrials)
        learner.ks = self.ks
        learner.lmbdas = self.lmbdas
        learner.u = self.u
        learner.folds = self.folds 
        learner.numAucSamples = self.numAucSamples
        
        return learner 

    def __str__(self): 
        outputStr = "WarpMf: lmbda=" + str(self.lmbda) + " k=" + str(self.k) + " alpha=" + str(self.alpha) + " batchSize=" + str(self.batchSize) + " maxTrials=" + str(self.maxTrials)
        outputStr += " numAucSamples=" + str(self.numAucSamples) + " folds=" + str(self.folds)
        
        return outputStr         