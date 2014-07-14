"""
Wrapper to use gamboviol implementation of CLiMF.
"""

import numpy 
import scipy.sparse 
import random
import sppy
import logging
import multiprocessing
import itertools
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import  MCEvaluator
from sandbox.recommendation.RecommenderUtils import computeTestF1
from sandbox.recommendation.AbstractRecommender import AbstractRecommender


try:
    from climf import _make_dataset
    from climf_fast import safe_climf_fast
except:
    logging.warning("climf not installed, cannot be used later on")
    def _make_dataset(X):
        raise NameError("climf is not installed, so '_make_dataset' is not defined")
    def safe_climf_fast(X,U,V,lbda,gamma,dim,max_iters,shuffle,seed,train_sample_users,sample_user_data):
        raise NameError("climf is not installed, so 'safe_climf_fast' is not defined")
        
        
class CLiMF(AbstractRecommender): 
    """
    An interface to use CLiMF recommender system. 
    """
    
    def __init__(self, k, lmbda, gamma, numProcesses=None): 
        super(CLiMF, self).__init__(numProcesses)
        self.k = k 
        self.lmbda = lmbda
        self.gamma = gamma
                
        self.max_iters = 25
        
        #Model selection parameters 
        self.ks = 2**numpy.arange(3, 8)
        self.lmbdas = 2.0**-numpy.arange(1, 20, 4)
        self.gammas = 2.0**-numpy.arange(1, 20, 4)
        self.verbose=1
        
    def initUV(self, X, k=None):
        if k == None:
            k=self.k 
        U = 0.01*numpy.random.random_sample((X.shape[0],k))
        V = 0.01*numpy.random.random_sample((X.shape[1],k))
        return U,V

    def learnModel(self, X, U=None, V=None):
        data = _make_dataset(X)
        if U==None or V==None:
            self.U, self.V = self.initUV(X)
        else:
            self.U = U
            self.V = V

        num_train_sample_users = X.shape[0]
        train_sample_users = numpy.array(random.sample(xrange(X.shape[0]),num_train_sample_users), dtype=numpy.int32)
        sample_user_data = numpy.array([numpy.array(X.getrow(i).indices, dtype=numpy.int32) for i in train_sample_users])
        
        safe_climf_fast(data, self.U, self.V, self.lmbda, self.gamma, self.U.shape[1], 
                   self.max_iters, False, 0, train_sample_users, sample_user_data, self.verbose)
    
    def predict(self, maxItems): 
        return MCEvaluator.recommendAtk(self.U, self.V, maxItems)
    
    def modelSelect(self, X, colProbs=None):
        """
        Perform model selection on X and return the best parameters. 
        """
        m, n = X.shape

        trainTestXs = Sampling.shuffleSplitRows(X, self.folds, self.validationSize, csarray=False, colProbs=colProbs)
        datas = []
        for (trainX, testX) in trainTestXs:
            testOmegaList = SparseUtils.getOmegaList(testX)
            testX = trainX+testX
            datas.append((trainX, testX, testOmegaList))
        testAucs = numpy.zeros((len(self.ks), len(self.lmbdas), len(self.gammas), len(trainTestXs)))
        
        logging.debug("Performing model selection")
        paramList = []        
        
        for i, k in enumerate(self.ks): 
            U, V = self.initUV(X, k)
            for lmbda in self.lmbdas:
                for gamma in self.gammas:
                    for (trainX, testX, testOmegaList) in datas:
                        learner = self.copy()
                        learner.k = k
                        learner.U = U.copy()
                        learner.V = V.copy()
                        learner.lmbda = lmbda
                        learner.gamma = gamma
                    
                        paramList.append((scipy.sparse.csr_matrix(trainX, dtype=numpy.float64), scipy.sparse.csr_matrix(testX, dtype=numpy.float64), learner))
            
        if self.numProcesses != 1: 
            pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
            resultsIterator = pool.imap(computeTestF1, paramList, self.chunkSize)
        else: 
            resultsIterator = itertools.imap(computeTestF1, paramList)
        
        for i_k in range(len(self.ks)):
            for i_lmbda in range(len(self.lmbdas)):
                for i_gamma in range(len(self.gammas)):
                    for i_cv in range(len(trainTestXs)):             
                        testAucs[i_k, i_lmbda, i_gamma, i_cv] = resultsIterator.next()
        
        if self.numProcesses != 1: 
            pool.terminate()
        
        meanTestMetrics = numpy.mean(testAucs, 3)
        stdTestMetrics = numpy.std(testAucs, 3)
        
        logging.debug("ks=" + str(self.ks))
        logging.debug("lmbdas=" + str(self.lmbdas))
        logging.debug("gammas=" + str(self.gammas))
        logging.debug("Mean metrics=" + str(meanTestMetrics))
        
        i_k, i_lmbda, i_gamma = numpy.unravel_index(meanTestMetrics.argmax(), meanTestMetrics.shape)
        self.k = self.ks[i_k]
        self.lmbda = self.lmbdas[i_lmbda]
        self.gamma = self.gammas[i_gamma]

        logging.debug("Model parameters: k=" + str(self.k) + " lmbda=" + str(self.lmbda) + " gamma=" + str(self.gamma))
         
        return meanTestMetrics, stdTestMetrics

            
    def copy(self): 
        learner = CLiMF(self.k, self.lmbda, self.gamma)
        self.copyParams(learner)
        learner.max_iters = self.max_iters
        learner.verbose = self.verbose
        
        return learner 

    def __str__(self): 
        outputStr = "CLiMF Recommender: k=" + str(self.k) 
        outputStr += " lambda=" + str(self.lmbda)
        outputStr += " gamma=" + str(self.gamma)
        outputStr += " max iters=" + str(self.max_iters)
        outputStr += super(CLiMF, self).__str__()
        
        return outputStr         

def main():
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    matrixFileName = PathDefaults.getDataDir() + "movielens/ml-100k/u.data" 
    data = numpy.loadtxt(matrixFileName)
    X = sppy.csarray((numpy.max(data[:, 0]), numpy.max(data[:, 1])), storagetype="row")
    X[data[:, 0]-1, data[:, 1]-1] = numpy.array(data[:, 2]>3, numpy.int)
    logging.debug("Read file: " + matrixFileName)
    logging.debug("Shape of data: " + str(X.shape))
    logging.debug("Number of non zeros " + str(X.nnz))
    
    u = 0.1 
    w = 1-u
    (m, n) = X.shape

    validationSize = 5
    trainTestXs = Sampling.shuffleSplitRows(X, 1, validationSize)
    trainX, testX = trainTestXs[0]
    trainX = trainX.toScipyCsr()

    learner = CLiMF(k=20, lmbda=0.001, gamma=0.0001)
    learner.learnModel(trainX)
    
if __name__=='__main__':
    main()


