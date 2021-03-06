
import numpy 
import logging
import multiprocessing 
from sandbox.recommendation.RecommenderUtils import computeTestMRR, computeTestF1
from sandbox.util.Sampling import Sampling 
from mrec.mf.wrmf import WRMFRecommender
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.recommendation.AbstractRecommender import AbstractRecommender


class WeightedMf(AbstractRecommender): 
    """
    A wrapper for the mrec class WRMFRecommender. 
    """
    
    def __init__(self, k, alpha=1, lmbda=0.015, maxIterations=20, w=0.9, numProcesses=None):
        super(WeightedMf, self).__init__(numProcesses)
        self.k = k 
        self.alpha = alpha
        #lmbda doesn't seem to make much difference at all 
        self.lmbda = lmbda 
        self.maxIterations = maxIterations 
        
        self.ks = 2**numpy.arange(3, 8)
        self.lmbdas = 2.0**-numpy.arange(-1, 12, 2)
    
        
    def learnModel(self, X): 
        
        learner = WRMFRecommender(self.k, self.alpha, self.lmbda, self.maxIterations)
        
        learner.fit(X)
        self.U = learner.U 
        self.V = learner.V 
        
        return self.U, self.V 

    def predict(self, maxItems): 
        return MCEvaluator.recommendAtk(self.U, self.U, maxItems)
        
    def modelSelect(self, X, colProbs=None): 
        """
        Perform model selection on X and return the best parameters. 
        """
        m, n = X.shape
        #cvInds = Sampling.randCrossValidation(self.folds, X.nnz)
        trainTestXs = Sampling.shuffleSplitRows(X, self.folds, self.validationSize, colProbs=colProbs)
        testMetrics = numpy.zeros((self.ks.shape[0], self.lmbdas.shape[0], len(trainTestXs)))
        
        if self.metric == "mrr":
            evaluationMethod = computeTestMRR
        elif self.metric == "f1": 
            evaluationMethod = computeTestF1
        else: 
            raise ValueError("Invalid metric: " + self.metric)        
        
        logging.debug("Performing model selection")
        paramList = []        
        
        for i, k in enumerate(self.ks): 
            for j, lmbda in enumerate(self.lmbdas): 
                for icv, (trainX, testX) in enumerate(trainTestXs):                
                    learner = self.copy()
                    learner.k = k
                    learner.lmbda = lmbda 
                
                    paramList.append((trainX.toScipyCsr(), testX.toScipyCsr(), learner))
            
        if self.numProcesses != 1: 
            pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
            resultsIterator = pool.imap(evaluationMethod, paramList, self.chunkSize)
        else: 
            import itertools
            resultsIterator = itertools.imap(evaluationMethod, paramList)
        
        for i, k in enumerate(self.ks):
            for j, lmbda in enumerate(self.lmbdas):
                for icv in range(len(trainTestXs)):             
                    testMetrics[i, j, icv] = resultsIterator.next()
        
        if self.numProcesses != 1: 
            pool.terminate()
            
        meanTestMetrics= numpy.mean(testMetrics, 2)
        stdTestMetrics = numpy.std(testMetrics, 2)
        
        logging.debug("ks=" + str(self.ks)) 
        logging.debug("lmbdas=" + str(self.lmbdas)) 
        logging.debug("Mean metrics=" + str(meanTestMetrics))
        
        self.k = self.ks[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[0]]
        self.lmbda = self.lmbdas[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[1]]

        logging.debug("Model parameters: k=" + str(self.k) + " lmbda=" + str(self.lmbda))
         
        return meanTestMetrics, stdTestMetrics
    
    def copy(self): 
        learner = WeightedMf(self.k, self.alpha, self.lmbda, self.maxIterations)
        self.copyParams(learner)
        learner.ks = self.ks
        learner.lmbdas = self.lmbdas

        return learner 

    def __str__(self): 
        outputStr = "WeightedMf: lmbda=" + str(self.lmbda) + " k=" + str(self.k) + " alpha=" + str(self.alpha)
        outputStr += " maxIterations=" + str(self.maxIterations)
        outputStr += super(WeightedMf, self).__str__()
        
        return outputStr         