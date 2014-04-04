
import numpy 
import logging
import multiprocessing 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.recommendation.MaxLocalAUCCython import  localAUCApprox
from sandbox.util.Sampling import Sampling 
from sandbox.util.Util import Util 
from mrec.mf.wrmf import WRMFRecommender
from sandbox.util.MCEvaluator import MCEvaluator 

def localAucsLmbdas(args): 
    trainX, testX, testOmegaList, learner  = args 
    
    (m, n) = trainX.shape
                        
    localAucs = numpy.zeros(learner.lmbdas.shape[0])

    for j, lmbda in enumerate(learner.lmbdas): 
        learner.lmbda = lmbda 
        
        U, V = learner.learnModel(trainX)
        
        r = SparseUtilsCython.computeR(U, V, 1-learner.u, learner.numAucSamples)
        localAucs[j] = localAUCApprox(testX, U, V, testOmegaList, learner.numAucSamples, r) 
        logging.debug("Local AUC: " + str(localAucs[j]) + " with k = " + str(learner.k) + " and lmbda= " + str(learner.lmbda))
        
    return localAucs

class WeightedMf(object): 
    """
    A wrapper for the mrec class WRMFRecommender. 
    """
    
    def __init__(self, k, alpha=1, lmbda=0.015, numIterations=15, u=0.1): 
        self.k = k 
        self.alpha = alpha 
        self.lmbda = lmbda 
        self.numIterations = numIterations 
        
        self.ks = numpy.array([10, 20, 50, 100])
        self.lmbdas = numpy.flipud(numpy.logspace(-3, -1, 11)*2) 
        
        self.folds = 3
        self.u = u
        self.numAucSamples = 100
        self.numProcesses = multiprocessing.cpu_count()
        self.chunkSize = 1
        
    def learnModel(self, X): 
        
        learner = WRMFRecommender(self.k, self.alpha, self.lmbda, self.numIterations)
        
        learner.fit(X)
        self.U = learner.U 
        self.V = learner.V 
        
        return self.U, self.V 

    def predict(self, maxItems): 
        return MCEvaluator.recommendAtk(self.U, self.U, maxItems)
        
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
        
        logging.debug("ks=" + str(self.ks)) 
        logging.debug("lmbdas=" + str(self.lmbdas)) 
        logging.debug("Mean local AUCs=" + str(meanLocalAucs))
        
        k = self.ks[numpy.unravel_index(numpy.argmax(meanLocalAucs), meanLocalAucs.shape)[0]]
        lmbda = self.lmbdas[numpy.unravel_index(numpy.argmax(meanLocalAucs), meanLocalAucs.shape)[1]]
        
        logging.debug("Model parameters: k=" + str(k) + " lmbda=" + str(lmbda))
        
        self.k = k 
        self.lmbda = lmbda 
        
        return meanLocalAucs, stdLocalAucs
    
    def copy(self): 
        learner = WeightedMf(self.k, self.alpha, self.lmbda, self.numIterations)
        learner.ks = self.ks
        learner.lmbdas = self.lmbdas
        learner.u = self.u
        learner.folds = self.folds 
        learner.numAucSamples = self.numAucSamples
        
        return learner 

    def __str__(self): 
        outputStr = "WeightedMf: lmbda=" + str(self.lmbda) + " k=" + str(self.k) + " alpha=" + str(self.alpha)
        outputStr += " numAucSamples=" + str(self.numAucSamples) + " numIterations=" + str(self.numIterations)
        
        return outputStr         