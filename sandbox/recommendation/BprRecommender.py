"""
Wrapper to use gamboviol implementation of BPR.
"""

import numpy 
import logging
import multiprocessing
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.Sampling import Sampling
from bpr import BPRArgs, BPR, UniformPairWithoutReplacement 
from bprCython import UniformUserUniformItem
from sandbox.recommendation.RecommenderUtils import computeTestMRR, computeTestF1
from sandbox.recommendation.AbstractRecommender import AbstractRecommender

class BprRecommender(AbstractRecommender): 
    """
    An interface to the BPR recommender system. 
    """
    
    def __init__(self, k, lmbdaUser=0.1, lmbdaPos=0.1, lmbdaNeg=0.1, biasReg=0.1, gamma=0.1, numProcesses=None): 
        """
        k is the number of factors, lambda is the regularistion and gamma is the learning rate 
        """
        super(BprRecommender, self).__init__(numProcesses)
        self.k = k 
        self.lmbdaUser = lmbdaUser
        self.lmbdaPos = lmbdaPos
        self.lmbdaNeg = lmbdaNeg
        self.biasReg = biasReg
        self.gamma = gamma
                
        self.maxIterations = 25
        
        #Model selection parameters 
        self.ks = 2**numpy.arange(3, 8)
        self.lmbdaUsers = 2.0**-numpy.arange(1, 20, 4)
        #self.lmbdaPoses = 2.0**-numpy.arange(1, 20, 4)
        #self.lmbdaNegs = 2.0**-numpy.arange(1, 20, 4)
        self.lmbdaItems = 2.0**-numpy.arange(1, 20, 4)
        self.gammas = 2.0**-numpy.arange(1, 20, 4)
        
    def learnModel(self, X, U=None, V=None):
        args = BPRArgs()
        args.learning_rate = self.gamma
        args.bias_regularization = self.biasReg
        args.user_regularization = self.lmbdaUser
        args.negative_item_regularization = self.lmbdaPos 
        args.positive_item_regularization = self.lmbdaNeg
        
        model = BPR(self.k, args)
    
        sample_negative_items_empirically = True
        sampler = UniformUserUniformItem(sample_negative_items_empirically)
        model.train(X, sampler, self.maxIterations)
        
        self.U = model.user_factors
        self.V = model.item_factors
    
    def predict(self, maxItems): 
        return MCEvaluator.recommendAtk(self.U, self.V, maxItems)
        
    def modelSelect(self, X, colProbs=None): 
        """
        Perform model selection on X and return the best parameters. 
        """
        m, n = X.shape
        trainTestXs = Sampling.shuffleSplitRows(X, self.folds, self.validationSize, csarray=False, colProbs=colProbs)
        testAucs = numpy.zeros((self.ks.shape[0], self.lmbdaUsers.shape[0], self.lmbdaItems.shape[0], len(trainTestXs)))
        
        logging.debug("Performing model selection with test leave out per row of " + str(self.validationSize))
        paramList = []        
        
        for i, k in enumerate(self.ks): 
            for j, lmbdaUser in enumerate(self.lmbdaUsers): 
                for s, lmbdaItem in enumerate(self.lmbdaItems): 
                        for icv, (trainX, testX) in enumerate(trainTestXs):
                            learner = self.copy()
                            learner.k = k  
                            learner.lmbdaUser = lmbdaUser 
                            learner.lmbdaPos = lmbdaItem
                            learner.lmbdaNeg = lmbdaItem
                        
                            paramList.append((trainX, testX, learner))
            
        if self.numProcesses != 1: 
            pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
            resultsIterator = pool.imap(computeTestF1, paramList, self.chunkSize)
        else: 
            import itertools
            resultsIterator = itertools.imap(computeTestF1, paramList)
        
        for i, k in enumerate(self.ks): 
            for j, lmbdaUser in enumerate(self.lmbdaUsers): 
                for s, lmbdaPos in enumerate(self.lmbdaItems): 
                        for icv, (trainX, testX) in enumerate(trainTestXs):        
                            testAucs[i, j, s, icv] = resultsIterator.next()
                
        if self.numProcesses != 1: 
            pool.terminate()
        
        meanTestLocalAucs = numpy.mean(testAucs, 3)
        stdTestLocalAucs = numpy.std(testAucs, 3)
        
        logging.debug("ks=" + str(self.ks)) 
        logging.debug("lmbdaUsers=" + str(self.lmbdaUsers)) 
        logging.debug("lmbdaPoses=" + str(self.lmbdaItems)) 
        logging.debug("lmbdaNegs=" + str(self.lmbdaItems)) 
        logging.debug("Mean local AUCs=" + str(meanTestLocalAucs))
        
        self.k = self.ks[numpy.unravel_index(numpy.argmax(meanTestLocalAucs), meanTestLocalAucs.shape)[0]]
        self.lmbdaUser = self.lmbdaUsers[numpy.unravel_index(numpy.argmax(meanTestLocalAucs), meanTestLocalAucs.shape)[1]]
        self.lmbdaPos = self.lmbdaItems[numpy.unravel_index(numpy.argmax(meanTestLocalAucs), meanTestLocalAucs.shape)[2]]
        self.lmbdaNeg = self.lmbdaItems[numpy.unravel_index(numpy.argmax(meanTestLocalAucs), meanTestLocalAucs.shape)[2]]

        logging.debug("Model parameters: " + str(self))
         
        return meanTestLocalAucs, stdTestLocalAucs
        
    def copy(self): 
        learner = BprRecommender(self.k, self.lmbdaUser, self.lmbdaPos, self.lmbdaNeg, self.biasReg, self.gamma)
        self.copyParams(learner)
        learner.maxIterations = self.maxIterations 
        
        return learner 

    def __str__(self): 
        outputStr = "BPR Recommender: k=" + str(self.k) 
        outputStr += " lmbdaUser=" + str(self.lmbdaUser)
        outputStr += " lmbdaPos=" + str(self.lmbdaPos)
        outputStr += " lmbdaNeg=" + str(self.lmbdaNeg)
        outputStr += " biasReg=" + str(self.biasReg)
        outputStr += " gamma=" + str(self.gamma)
        outputStr += " maxIterations=" + str(self.maxIterations)
        outputStr += super(BprRecommender, self).__str__()
        
        return outputStr   