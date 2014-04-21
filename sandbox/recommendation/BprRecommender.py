"""
Wrapper to use gamboviol implementation of BPR.
"""

import numpy 
import sppy
import logging
import multiprocessing
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.recommendation.MaxLocalAUCCython import localAUCApprox
from bpr import BPRArgs, BPR, UniformPairWithoutReplacement 


def computeTestAuc(args): 
    trainX, testX, learner  = args 
    
    allX = trainX + testX
    testOmegaList = SparseUtils.getOmegaList(testX)
    logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
    
    learner.learnModel(trainX)
    
    U = learner.U 
    V = learner.V 
    
    #r = SparseUtilsCython.computeR(U, V, learner.w, learner.numRecordAucSamples)
    testAuc = MCEvaluator.localAUCApprox(allX, U, V, learner.w, learner.numRecordAucSamples, testOmegaList)
    logging.debug("Weighted local AUC: " + str(testAuc) + " with k=" + str(learner.k) + " lmbda=" + str(learner.lmbda))
        
    return testAuc

class BprRecommender(object): 
    """
    An interface to the BPR recommender system. 
    """
    
    def __init__(self, k, lmbda, gamma, w=0.9): 
        """
        k is the number of factors, lambda is the regularistion and gamma is the learning rate 
        """
        self.k = k 
        self.lmbda = lmbda
        self.gamma = gamma
                
        self.maxIterations = 25
        
        #Model selection parameters 
        self.folds = 5 
        self.testSize = 3
        self.ks = 2**numpy.arange(3, 8)
        self.lmbdas = 2.0**-numpy.arange(1, 20, 4)
        self.gammas = 2.0**-numpy.arange(1, 20, 4)
        
        
        self.numRecordAucSamples = 500
        self.w = w
        
        self.numProcesses = multiprocessing.cpu_count()
        self.chunkSize = 1
        

    def learnModel(self, X, U=None, V=None):
        args = BPRArgs()
        args.learning_rate = self.gamma
        #args.bias_regularization = self.lmbda
        args.negative_item_regularization = self.lmbda 
        args.positive_item_regularization = self.lmbda
        
        model = BPR(self.k, args)
    
        sample_negative_items_empirically = True
        sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
        model.train(X, sampler, self.maxIterations)
        
        self.U = model.user_factors
        self.V = model.item_factors
    
    def predict(self, maxItems): 
        return MCEvaluator.recommendAtk(self.U, self.V, maxItems)
        
    def modelSelect(self, X): 
        """
        Perform model selection on X and return the best parameters. 
        """
        m, n = X.shape
        #cvInds = Sampling.randCrossValidation(self.folds, X.nnz)
        trainTestXs = Sampling.shuffleSplitRows(X, self.folds, self.testSize, csarray=False)
        testAucs = numpy.zeros((self.ks.shape[0], self.lmbdas.shape[0], len(trainTestXs)))
        
        logging.debug("Performing model selection with test leave out per row of " + str(self.testSize))
        paramList = []        
        
        for i, k in enumerate(self.ks): 
            for j, lmbda in enumerate(self.lmbdas): 
                for icv, (trainX, testX) in enumerate(trainTestXs):
                    learner = self.copy()
                    learner.k = k  
                    learner.lmbda = lmbda 
                
                    paramList.append((trainX, testX, learner))
            
        if self.numProcesses != 1: 
            pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
            resultsIterator = pool.imap(computeTestAuc, paramList, self.chunkSize)
        else: 
            import itertools
            resultsIterator = itertools.imap(computeTestAuc, paramList)
        
        for i, k in enumerate(self.ks): 
            for j, lmbda in enumerate(self.lmbdas): 
                for icv, (trainX, testX) in enumerate(trainTestXs):         
                    testAucs[i, j, icv] = resultsIterator.next()
        
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
        learner = BprRecommender(self.k, self.lmbda, self.gamma)
        learner.maxIterations = self.maxIterations
        
        return learner 

    def __str__(self): 
        outputStr = "BPR Recommender: k=" + str(self.k) 
        outputStr += " lambda=" + str(self.lmbda)
        outputStr += " gamma=" + str(self.gamma)
        outputStr += " maxIterations=" + str(self.maxIterations)
        
        return outputStr   