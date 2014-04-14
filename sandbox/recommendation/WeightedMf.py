
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

def computeTestAucs(args): 
    trainX, testX, learner  = args 
    testOmegaList = SparseUtils.getOmegaList(testX)
    X = testX+trainX
    
    testAucScores = numpy.zeros(learner.lmbdas.shape[0])
    logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
    
    for i, lmbda in enumerate(learner.lmbdas):         
        learner.lmbda = lmbda 

        U, V = learner.learnModel(trainX)
        r = SparseUtilsCython.computeR(U, V, learner.w, learner.numRecordAucSamples)
        testAucScores[i] = localAUCApprox(X, U, V, testOmegaList, learner.numRecordAucSamples, r)
        
        logging.debug("Local AUC: " + str(testAucScores[i]) + " with k=" + str(learner.k) + " lmbda=" + str(learner.lmbda))
        
    return testAucScores

class WeightedMf(object): 
    """
    A wrapper for the mrec class WRMFRecommender. 
    """
    
    def __init__(self, k, alpha=1, lmbda=0.015, numIterations=15, w=0.9): 
        self.k = k 
        self.alpha = alpha 
        self.lmbda = lmbda 
        self.numIterations = numIterations 
        
        self.ks = numpy.array([10, 20, 50, 100])
        self.lmbdas = numpy.flipud(numpy.logspace(-3, -1, 11)*2) 
        
        self.folds = 3
        self.testSize = 3
        self.w = w
        self.numRecordAucSamples = 500
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
        #cvInds = Sampling.randCrossValidation(self.folds, X.nnz)
        trainTestXs = Sampling.shuffleSplitRows(X, self.folds, self.testSize)
        testAucs = numpy.zeros((self.ks.shape[0], self.lmbdas.shape[0], len(trainTestXs)))
        
        logging.debug("Performing model selection")
        paramList = []        
        
        for i, k in enumerate(self.ks): 
            self.k = k
            
            for icv, (trainX, testX) in enumerate(trainTestXs):                
                learner = self.copy()
                learner.k = k                
            
                paramList.append((trainX.toScipyCsr(), testX.toScipyCsr(), learner))
            
        if self.numProcesses != 1: 
            pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
            resultsIterator = pool.imap(computeTestAucs, paramList, self.chunkSize)
        else: 
            import itertools
            resultsIterator = itertools.imap(computeTestAucs, paramList)
        
        for i, k in enumerate(self.ks):
            for icv in range(len(trainTestXs)):             
                testAucs[i, :, icv] = resultsIterator.next()
        
        if self.numProcesses != 1: 
            pool.terminate()
        
        meanTestLocalAucs = numpy.mean(testAucs, 2)
        stdTestLocalAucs = numpy.std(testAucs, 2)
        
        logging.debug("ks=" + str(self.ks)) 
        logging.debug("lmbdas=" + str(self.lmbdas)) 
        logging.debug("Mean local AUCs=" + str(meanTestLocalAucs))
        
        self.k = self.ks[numpy.unravel_index(numpy.argmax(meanTestLocalAucs), meanTestLocalAucs.shape)[0]]
        self.lmbda = self.lmbdas[numpy.unravel_index(numpy.argmax(meanTestLocalAucs), meanTestLocalAucs.shape)[1]]

        logging.debug("Model parameters: k=" + str(self.k) + " lmbda=" + str(self.lmbda))
         
        return meanTestLocalAucs, stdTestLocalAucs
    
    def copy(self): 
        learner = WeightedMf(self.k, self.alpha, self.lmbda, self.numIterations)
        learner.ks = self.ks
        learner.lmbdas = self.lmbdas
        learner.w = self.w
        learner.folds = self.folds 
        learner.numRecordAucSamples = self.numRecordAucSamples
        
        return learner 

    def __str__(self): 
        outputStr = "WeightedMf: lmbda=" + str(self.lmbda) + " k=" + str(self.k) + " alpha=" + str(self.alpha)
        outputStr += " numRecordAucSamples=" + str(self.numRecordAucSamples) + " numIterations=" + str(self.numIterations)
        
        return outputStr         