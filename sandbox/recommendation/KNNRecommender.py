# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 15:59:28 2014

@author: charanpal
"""


import numpy 
import logging
import multiprocessing 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.util.Sampling import Sampling 
from sandbox.util.Util import Util 
from mrec.item_similarity.knn import CosineKNNRecommender
from sandbox.util.MCEvaluator import MCEvaluator 

def computePrecision(args): 
    trainX, testX, testOmegaList, learner  = args 
    
    (m, n) = trainX.shape
                                             
    learner.learnModel(trainX)
    maxItems = 20
    orderedItems = learner.predict(maxItems)
    #print(orderedItems)
    precision = MCEvaluator.precisionAtK(testX, orderedItems, maxItems)
        
    logging.debug("Precision@" + str(maxItems) + ": " + str(precision) + " with k = " + str(learner.k))
        
    return precision

class KNNRecommender(object): 
    """
    A wrapper for the mrec class CosineKNNRecommender. 
    """
    
    def __init__(self, k): 
        self.k = k 
                
        self.folds = 3
        self.numAucSamples = 100
        self.numProcesses = multiprocessing.cpu_count()
        self.chunkSize = 1
        
    def learnModel(self, X): 
        self.X = X 
        self.learner = CosineKNNRecommender(self.k)
        self.learner.fit(X)
    
    def predict(self, maxItems):
        orderedItems = self.learner.batch_recommend_items(self.X, maxItems, return_scores=False)
        orderedItems = numpy.array(orderedItems)
        return orderedItems 
        
    def modelSelect(self, X): 
        """
        Perform model selection on X and return the best parameters. 
        """
        m, n = X.shape
        cvInds = Sampling.randCrossValidation(self.folds, X.nnz)
        precisions = numpy.zeros((self.ks.shape[0], len(cvInds)))
        
        logging.debug("Performing model selection")
        paramList = []        
        
        for icv, (trainInds, testInds) in enumerate(cvInds):
            Util.printIteration(icv, 1, self.folds, "Fold: ")

            trainX = SparseUtils.submatrix(X, trainInds)
            testX = SparseUtils.submatrix(X, testInds)
            
            testOmegaList = SparseUtils.getOmegaList(testX)
            
            for i, k in enumerate(self.ks): 
                learner = self.copy()
                learner.k = k
                paramList.append((trainX, testX, testOmegaList, learner))
                    
        #pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
        #resultsIterator = pool.imap(computePrecision, paramList, self.chunkSize)
        import itertools
        resultsIterator = itertools.imap(computePrecision, paramList)
        
        for icv, (trainInds, testInds) in enumerate(cvInds):        
            for i, k in enumerate(self.ks): 
                tempPrecision = resultsIterator.next()
                precisions[i, icv] = tempPrecision
        
        #pool.terminate()
        
        meanPrecisions = numpy.mean(precisions, 1)
        stdPrecisions = numpy.std(precisions, 1)
        
        logging.debug(meanPrecisions)
        
        k = self.ks[numpy.argmax(meanPrecisions)]

        
        logging.debug("Model parameters: k=" + str(k)) 
        
        self.k = k 
        
        return meanPrecisions, stdPrecisions
    
    def copy(self): 
        learner = KNNRecommender(self.k)
        learner.ks = self.ks
        learner.folds = self.folds 
        learner.numAucSamples = self.numAucSamples
        
        return learner 

    def __str__(self): 
        outputStr = "KnnRecommender: k=" + str(self.k) 
        outputStr += " numAucSamples=" + str(self.numAucSamples)
        
        return outputStr         