
import numpy 
import logging
import multiprocessing 
import sppy 
import time
import scipy.sparse
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.recommendation.MaxLocalAUCCython import derivativeUi, derivativeVi, updateUVApprox, objectiveApprox, updateV, updateU
from sandbox.util.Sampling import Sampling 
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython 
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute 
from sandbox.recommendation.WeightedMf import WeightedMf
from sandbox.recommendation.RecommenderUtils import computeTestPrecision, computeTestF1
from sandbox.misc.RandomisedSVD import RandomisedSVD
from sandbox.recommendation.AbstractRecommender import AbstractRecommender

def computeObjective(args): 
    """
    Compute the objective for a particular parameter set. Used to set a learning rate. 
    """
    X, U, V, maxLocalAuc = args 
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.singleLearnModel(X, U=U, V=V, verbose=True)
    obj = trainMeasures[-1, 0]
    logging.debug("Final objective: " + str(obj) + " with t0=" + str(maxLocalAuc.t0) + " and alpha=" + str(maxLocalAuc.alpha))
    return obj
    
def computeTestAuc(args): 
    trainX, testX, U, V, maxLocalAuc = args 
    
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.singleLearnModel(trainX, U=U, V=V, verbose=True)
    testAucs = testMeasures[:, 1]
    muAuc = numpy.average(testAucs, weights=numpy.flipud(1/numpy.arange(1, len(testAucs)+1, dtype=numpy.float)))
    logging.debug("Weighted local AUC: " + str('%.4f' % muAuc) + " with k=" + str(maxLocalAuc.k) + " lmbda=" + str(maxLocalAuc.lmbda) + " rho=" + str(maxLocalAuc.rho))
        
    return muAuc
      
def computeFactors(args): 
    """
    Compute the objective for a particular parameter set. Used to set a learning rate. 
    """
    X, U, V, maxLocalAuc = args 
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.singleLearnModel(X, U=U, V=V, verbose=True)
    return U, V, trainMeasures, testMeasures, iterations, totalTime      
      
class MaxLocalAUC(AbstractRecommender): 
    def __init__(self, k, w, alpha=0.05, eps=10**-6, lmbdaU=1, lmbdaV=1, stochastic=False, numProcesses=None): 
        """
        Create an object for  maximising the local AUC with a penalty term using the matrix
        decomposition UV.T 
                
        :param k: The rank of matrices U and V
        
        :param w: The quantile for the local AUC - e.g. 1 means takes the largest value, 0.7 means take the top 0.3 
        
        :param alpha: The (initial) learning rate 
        
        :param eps: The termination threshold for ||dU|| and ||dV||
        
        :param lmbda: The regularistion penalty for V 
        
        :stochastic: Whether to use stochastic gradient descent or gradient descent 
        """
        super(MaxLocalAUC, self).__init__(numProcesses)
        self.k = k 
        self.w = w
        self.eps = eps 
        self.stochastic = stochastic
        self.parallelSGD = False

        self.rate = "constant"
        self.alpha = alpha #Initial learning rate 
        self.t0 = 0.1 #Convergence speed - larger means we get to 0 faster
        self.beta = 0.75
        
        self.normalise = True
        self.lmbdaU = lmbdaU 
        self.lmbdaV = lmbdaV 
        self.rho = 1.00 #Penalise low rank elements 
        self.itemExpP = 0.0 #Sample from power law between 0 and 1 
        self.itemExpQ = 0.5        
        
        self.recordStep = 10
        self.numRowSamples = 100
        self.numAucSamples = 10
        self.numRecordAucSamples = 100
        #1 iterations is a complete run over the dataset (i.e. m gradients)
        self.maxIterations = 50
        self.initialAlg = "rand"
        #Possible choices are uniform, top, rank 
        self.sampling = "uniform"
        
        #Model selection parameters 
        self.validationUsers = 0.1
        self.ks = 2**numpy.arange(3, 8)
        self.lmbdas = numpy.linspace(0.5, 2.0, 7)
        self.rhos = numpy.array([0, 0.1, 0.5, 1.0])
        self.metric = "auc"

        #Learning rate selection 
        self.t0s = 10**-numpy.arange(2, 5, 0.5)
        self.alphas = 2.0**-numpy.arange(-1, 3, 0.5)
    
    def recordResults(self, muU, muV, trainMeasures, testMeasures, sigma, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, gi, gp, gq, trainX): 
        r = SparseUtilsCython.computeR(muU, muV, self.w, self.numRecordAucSamples)
        objArr = self.objectiveApprox((indPtr, colInds), muU, muV, r, gi, gp, gq, full=True)
        trainMeasures.append([objArr.sum(), MCEvaluator.localAUCApprox((indPtr, colInds), muU, muV, self.w, self.numRecordAucSamples, r)]) 
        
        testMeasuresRow = []
        testMeasuresRow.append(self.objectiveApprox((testIndPtr, testColInds), muU, muV, r, gi, gp, gq, allArray=(allIndPtr, allColInds)))
        testMeasuresRow.append(MCEvaluator.localAUCApprox((testIndPtr, testColInds), muU, muV, self.w, self.numRecordAucSamples, r, allArray=(allIndPtr, allColInds)))
        testOrderedItems = MCEvaluatorCython.recommendAtk(muU, muV, self.recommendSize, trainX)
        f1Array, orderedItems = MCEvaluator.f1AtK((testIndPtr, testColInds), testOrderedItems, self.recommendSize, verbose=True)
        testMeasuresRow.append(f1Array[rowSamples].mean())   
        mrr = MCEvaluatorCython.reciprocalRankAtk(testIndPtr, testColInds, testOrderedItems)
        testMeasuresRow.append(mrr[rowSamples].mean())
        testMeasures.append(testMeasuresRow)
           
        printStr = "iter " + str(loopInd) + ":"
        printStr += " sigma=" + str('%.4f' % sigma)
        printStr += " train: obj~" + str('%.4f' % trainMeasures[-1][0]) 
        printStr += " LAUC~" + str('%.4f' % trainMeasures[-1][1]) 
        printStr += " validation: obj~" + str('%.4f' % testMeasuresRow[0])
        printStr += " LAUC~" + str('%.4f' % testMeasuresRow[1])
        printStr += " f1@" + str(self.recommendSize) + "=" + str('%.4f' % testMeasuresRow[2])
        printStr += " mrr@" + str(self.recommendSize) + "=" + str('%.4f' % testMeasuresRow[3])
        printStr += " ||U||=" + str('%.3f' % numpy.linalg.norm(muU))
        printStr += " ||V||=" + str('%.3f' %  numpy.linalg.norm(muV))
        
        return printStr
    
    def singleLearnModel(self, X, verbose=False, U=None, V=None): 
        """
        Max local AUC with Frobenius norm penalty on V. Solve with (stochastic) gradient descent. 
        The input is a sparse array. 
        """
        #Convert to a csarray for faster access 
        if scipy.sparse.issparse(X):
            logging.debug("Converting to csarray")
            X2 = sppy.csarray(X, storagetype="row")
            X = X2        
        
        #We keep a validation set in order to determine when to stop 
        numValidationUsers = max(int(X.shape[0]*self.validationUsers), 2)
        trainX, testX, rowSamples = Sampling.shuffleSplitRows(X, 1, self.validationSize, numRows=numValidationUsers)[0]  

        m = trainX.shape[0]
        n = trainX.shape[1]
        indPtr, colInds = SparseUtils.getOmegaListPtr(trainX)
        
        #Not that to compute the test AUC we pick i \in X and j \notin X \cup testX        
        testIndPtr, testColInds = SparseUtils.getOmegaListPtr(testX)
        allIndPtr, allColInds = SparseUtils.getOmegaListPtr(X)

        if U==None or V==None:
            U, V = self.initUV(trainX)
        
        sigma = self.alpha
        
        muU = U.copy() 
        muV = V.copy()
        
        #Store best results 
        bestF1 = 0 
        bestU = 0 
        bestV = 0
        
        trainMeasures = []
        testMeasures = []        
    
        loopInd = 0
        
        #Set up order of indices for stochastic methods 
        permutedRowInds = numpy.array(numpy.random.permutation(m), numpy.uint32)
        permutedColInds = numpy.array(numpy.random.permutation(n), numpy.uint32)
        
        startTime = time.time()
        #self.wv = 1 - X.sum(1)/float(n)
        
        #uniform weights 
        gi = numpy.ones(m)/float(m)
        #gp = numpy.ones(n)/float(n)
        #gq = numpy.ones(n)/float(n)
        
        itemProbs = (X.sum(0)+1)/float(m+1)
        gp = itemProbs**self.itemExpP 
        gp /= gp.sum()
        #gq = numpy.ones(n)
        gq = (1-itemProbs)**self.itemExpQ 
        #gq = (itemProbs)**self.itemExpQ
        gq /= gq.sum()
        
        normGp, normGq = self.computeNormGpq(indPtr, colInds, gp, gq, m)
        
        #Make user weights sum of gp[i] for all i 
        """
        gi = numpy.zeros(m)
        for i in range(m): 
            omegai = allColInds[allIndPtr[i]:allIndPtr[i+1]]
            
            for j in omegai:
                gi[i] += gp[j]
                
        gi /= gi.sum()
        """ 
    
        while loopInd < self.maxIterations:           
            if self.rate == "constant": 
                sigma = self.alpha 
            elif self.rate == "optimal":
                sigma = self.alpha/((1 + self.alpha*self.t0*loopInd**self.beta))
            else: 
                raise ValueError("Invalid rate: " + self.rate)
            
            if loopInd % self.recordStep == 0: 
   
                printStr = self.recordResults(muU, muV, trainMeasures, testMeasures, sigma, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, gi, gp, gq, trainX)
                               
                if loopInd != 0: 
                    print("")  
                    
                logging.debug(printStr) 
                                
                if testMeasures[-1][2] >= bestF1: 
                    bestF1 = testMeasures[-1][2]
                    bestU = muU 
                    bestV = muV 
                
            U  = numpy.ascontiguousarray(U)
            
            self.updateUV(indPtr, colInds, U, V, muU, muV, permutedRowInds, permutedColInds, gi, gp, gq, normGp, normGq, loopInd, sigma)                       
                
            loopInd += 1
            
        #Compute quantities for last U and V 
        totalTime = time.time() - startTime
        printStr = "\nFinished, time=" + str('%.1f' % totalTime) + " "
        printStr += self.recordResults(muU, muV, trainMeasures, testMeasures, sigma, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, gi, gp, gq, trainX)
        logging.debug(printStr)
         
        self.U = bestU 
        self.V = bestV
        self.gi = gi
        self.gp = gp
        self.gq = gq
         
        if verbose:     
            return self.U, self.V, numpy.array(trainMeasures), numpy.array(testMeasures), loopInd, totalTime
        else: 
            return self.U, self.V
      
    def parallelLearnModel(self, X, verbose=False, U=None, V=None): 
        """
        Perform model learning in parallel. 
        """
        
        cvInds = Sampling.crossValidation(self.numProcesses, X.nnz)
        paramList = []
        
        for trainInds, testInds in cvInds: 
            testX = SparseUtils.submatrix(X, testInds)
            learner = self.copy()
            
            paramList.append((testX, U, V, learner))
        
        if self.numProcesses != 1: 
            pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
            resultsIterator = pool.imap(computeFactors, paramList, self.chunkSize)
        else: 
            import itertools
            resultsIterator = itertools.imap(computeFactors, paramList)
            
        self.U = numpy.zeros((X.shape[0], self.k))
        self.V  = numpy.zeros((X.shape[1], self.k))
       
        for k in range(self.numProcesses):
            tempU, tempV, trainMeasures, testMeasures, iterations, totalTime = resultsIterator.next()
            self.U += tempU 
            self.V += tempV 
            
        if self.numProcesses != 1: 
            pool.terminate()
            
        self.U  /= self.numProcesses
        self.V  /= self.numProcesses
        
        if verbose:
            return self.U , self.V, trainMeasures, testMeasures, iterations, totalTime 
        else: 
            return self.U , self.V 
        
    def learnModel(self, X, verbose=False, U=None, V=None): 
        if self.parallelSGD: 
            return self.parallelLearnModel(X, verbose, U, V)
        else: 
            return self.singleLearnModel(X, verbose, U, V)
        
    def predict(self, maxItems): 
        return MCEvaluator.recommendAtk(self.U, self.V, maxItems)
          
    def initUV(self, X): 
        m = X.shape[0]
        n = X.shape[1]        
        
        if self.initialAlg == "rand": 
            U = numpy.random.randn(m, self.k)*0.1
            V = numpy.random.randn(n, self.k)*0.1
        elif self.initialAlg == "svd":
            logging.debug("Initialising with Randomised SVD")
            U, s, V = RandomisedSVD.svd(X, self.k)
            U = U*s
        elif self.initialAlg == "softimpute": 
            logging.debug("Initialising with softimpute")
            trainIterator = iter([X.toScipyCsc()])
            rho = 0.01
            learner = IterativeSoftImpute(rho, k=self.k, svdAlg="propack", postProcess=True)
            ZList = learner.learnModel(trainIterator)    
            U, s, V = ZList.next()
            U = U*s
        elif self.initialAlg == "wrmf": 
            logging.debug("Initialising with wrmf")
            learner = WeightedMf(self.k, w=self.w)
            U, V = learner.learnModel(X.toScipyCsr())            
        else:
            raise ValueError("Unknown initialisation: " + str(self.initialAlg))  
         
        U = numpy.ascontiguousarray(U)
        #maxNorm = numpy.sqrt(numpy.max(numpy.sum(U**2, 1)))
        #U = U/maxNorm  
        
        V = numpy.ascontiguousarray(V) 
        #maxNorm = numpy.sqrt(numpy.max(numpy.sum(V**2, 1)))
        #V = V/maxNorm
        
        return U, V
        
    def updateUV(self, indPtr, colInds, U, V, muU, muV, permutedRowInds, permutedColInds, gi, gp, gq, normGp, normGq, ind, sigma): 
        """
        Find the derivative with respect to V or part of it. 
        """
        if not self.stochastic:               
            r = SparseUtilsCython.computeR(U, V, self.w, self.numRecordAucSamples)  
            #r = SparseUtilsCython.computeR2(U, V, self.wv, self.numRecordAucSamples)
            updateU(indPtr, colInds, U, V, r, sigma, self.rho, self.normalise)
            updateV(indPtr, colInds, U, V, r, sigma, self.lmbda, self.rho, self.normalise)
            
            muU[:] = U[:] 
            muV[:] = V[:]
        else: 
            
            updateUVApprox(indPtr, colInds, U, V, muU, muV, permutedRowInds, permutedColInds, gi, gp, gq, normGp, normGq, ind, sigma, self.numRowSamples, self.numAucSamples, self.w, self.lmbdaU, self.lmbdaV, self.rho, self.normalise)

    def computeNormGpq(self, indPtr, colInds, gp, gq, m):
        
        normGp = numpy.zeros(m)
        normGq = numpy.zeros(m) 
        gqSum = gq.sum()
        
        for i in range(m):
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            normGp[i] = gp[omegai].sum()
            normGq[i] = gqSum - gq[omegai].sum()
            
        return normGp, normGq
        
    def derivativeUi(self, indPtr, colInds, U, V, r, c, i): 
        """
        delta phi/delta u_i
        """
        return derivativeUi(indPtr, colInds, U, V, r, c, i, self.rho, self.normalise)
        
    def derivativeVi(self, X, U, V, omegaList, i, r, c): 
        """
        delta phi/delta v_i
        """
        return derivativeVi(X, U, V, omegaList, i, r, c, self.rho, self.normalise)           

    def omegaProbsUniform(self, indPtr, colInds, U, V): 
        """
        All positive items have the same probability. 
        """
        colIndsCumProbs = numpy.ones(colInds.shape[0])
        m = U.shape[0]
        
        for i in range(m):
            colIndsCumProbs[indPtr[i]:indPtr[i+1]] /= colIndsCumProbs[indPtr[i]:indPtr[i+1]].sum()
            colIndsCumProbs[indPtr[i]:indPtr[i+1]]  = numpy.cumsum(colIndsCumProbs[indPtr[i]:indPtr[i+1]])
            
        return colIndsCumProbs

    def omegaProbsTopZ(self, indPtr, colInds, U, V): 
        """
        For the set of positive items in each row, select the largest z and give 
        them equal probability, and the remaining items zero probability. 
        """
        colIndsCumProbs = numpy.zeros(colInds.shape[0])
        m = U.shape[0]
        
        for i in range(m):
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            uiVOmegai = U[i, :].T.dot(V[omegai, :].T)
            ri = numpy.sort(uiVOmegai)[-min(self.recommendSize, uiVOmegai.shape[0])]
            colIndsCumProbs[indPtr[i]:indPtr[i+1]] = numpy.array(uiVOmegai >= ri, numpy.float)
            colIndsCumProbs[indPtr[i]:indPtr[i+1]] /= colIndsCumProbs[indPtr[i]:indPtr[i+1]].sum()
            colIndsCumProbs[indPtr[i]:indPtr[i+1]]  = numpy.cumsum(colIndsCumProbs[indPtr[i]:indPtr[i+1]])
            
        return colIndsCumProbs
            
    def omegaProbsRank(self, indPtr, colInds, U, V): 
        """
        Take the positive values in each row and sort them according to their 
        values in U.V^T then give p(j) = ind(j)+1 where ind(j) is the index of the 
        jth item. 
        """
        colIndsCumProbs = numpy.zeros(colInds.shape[0])
        m = U.shape[0]
        
        for i in range(m):
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            uiVOmegai = U[i, :].T.dot(V[omegai, :].T)
            inds = numpy.argsort(numpy.argsort((uiVOmegai)))+1
            colIndsCumProbs[indPtr[i]:indPtr[i+1]] = inds/float(inds.sum())
            colIndsCumProbs[indPtr[i]:indPtr[i+1]]  = numpy.cumsum(colIndsCumProbs[indPtr[i]:indPtr[i+1]])
            
        return colIndsCumProbs


    def learningRateSelect(self, X): 
        """
        Let's set the initial learning rate. 
        """        
        m, n = X.shape
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
        objectives = numpy.zeros((self.t0s.shape[0], self.alphas.shape[0]))
        
        paramList = []   
        logging.debug("t0s=" + str(self.t0s))
        logging.debug("alphas=" + str(self.alphas))
        logging.debug(self)
        
        numInitalUVs = self.folds
            
        for k in range(numInitalUVs):
            U, V = self.initUV(X)
                        
            for i, t0 in enumerate(self.t0s): 
                for j, alpha in enumerate(self.alphas): 
                    maxLocalAuc = self.copy()
                    maxLocalAuc.t0 = t0
                    maxLocalAuc.alpha = alpha 
                    paramList.append((X, U, V, maxLocalAuc))
                    
        if self.numProcesses != 1: 
            pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
            resultsIterator = pool.imap(computeObjective, paramList, self.chunkSize)
        else: 
            import itertools
            resultsIterator = itertools.imap(computeObjective, paramList)
        
        for k in range(numInitalUVs):
            for i, t0 in enumerate(self.t0s): 
                for j, alpha in enumerate(self.alphas):  
                    objectives[i, j] += resultsIterator.next()
            
        if self.numProcesses != 1: 
            pool.terminate()
            
        objectives /= float(numInitalUVs)   
        logging.debug("t0s=" + str(self.t0s))
        logging.debug("alphas=" + str(self.alphas))
        logging.debug(objectives)
        
        t0 = self.t0s[numpy.unravel_index(numpy.argmin(objectives), objectives.shape)[0]]
        alpha = self.alphas[numpy.unravel_index(numpy.argmin(objectives), objectives.shape)[1]]
        
        logging.debug("Learning rate parameters: t0=" + str(t0) + " alpha=" + str(alpha))
        
        self.t0 = t0 
        self.alpha = alpha 
        
        return objectives
        
    def modelSelect(self, X, colProbs=None): 
        """
        Perform model selection on X and return the best parameters. 
        """
        m, n = X.shape
        trainTestXs = Sampling.shuffleSplitRows(X, self.folds, self.validationSize, colProbs=colProbs)
        testAucs = numpy.zeros((self.ks.shape[0], self.lmbdas.shape[0], self.rhos.shape[0], len(trainTestXs)))
        
        logging.debug("Performing model selection with test leave out per row of " + str(self.validationSize))
        paramList = []        
        
        for i, k in enumerate(self.ks): 
            self.k = k
            
            for icv, (trainX, testX) in enumerate(trainTestXs):
                U, V = self.initUV(trainX)
                for j, lmbda in enumerate(self.lmbdas): 
                    for s, rho in enumerate(self.rhos): 
                        maxLocalAuc = self.copy()
                        maxLocalAuc.k = k    
                        maxLocalAuc.lmbdaU = lmbda
                        maxLocalAuc.lmbdaV = lmbda
                        maxLocalAuc.rho = rho 
                    
                        paramList.append((trainX, testX, maxLocalAuc))
            
        logging.debug("Set parameters")
        if self.metric == "auc":
            evaluationMethod = computeTestAuc
        elif self.metric == "precision": 
            evaluationMethod = computeTestPrecision
        elif self.metric == "recall": 
            evaluationMethod = computeTestRecall
        elif self.metric == "f1": 
            evaluationMethod = computeTestF1
        else: 
            raise ValueError("Invalid metric: " + self.metric)
        
        if self.numProcesses != 1: 
            pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
            resultsIterator = pool.imap(evaluationMethod, paramList, self.chunkSize)
        else: 
            import itertools
            resultsIterator = itertools.imap(evaluationMethod, paramList)
        
        for i, k in enumerate(self.ks):
            for icv in range(len(trainTestXs)): 
                for j, lmbda in enumerate(self.lmbdas): 
                    for s, rho in enumerate(self.rhos): 
                        testAucs[i, j, s, icv] = resultsIterator.next()
        
        if self.numProcesses != 1: 
            pool.terminate()
        
        meanTestMetrics = numpy.mean(testAucs, 3)
        stdTestMetrics = numpy.std(testAucs, 3)
        
        logging.debug("ks=" + str(self.ks)) 
        logging.debug("lmbdas=" + str(self.lmbdas)) 
        logging.debug("rhos=" + str(self.rhos)) 
        logging.debug("Mean metrics =" + str(meanTestMetrics))
        
        self.k = self.ks[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[0]]
        self.lmbdaU = self.lmbdas[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[1]]
        self.lmbdaV = self.lmbdas[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[1]]
        self.rho = self.rhos[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[2]]

        logging.debug("Model parameters: k=" + str(self.k) + " lmbdaU=" + str(self.lmbdaU) + " lmbdaV=" + str(self.lmbdaV) + " rho=" + str(self.rho) +  " max=" + str(numpy.max(meanTestMetrics)))
         
        return meanTestMetrics, stdTestMetrics
  
    def objectiveApprox(self, positiveArray, U, V, r, gi, gp, gq, allArray=None, full=False): 
        """
        Compute the estimated local AUC for the score functions UV^T relative to X with 
        quantile w. The AUC is computed using positiveArray which is a tuple (indPtr, colInds)
        assuming allArray is None. If allArray is not None then positive items are chosen 
        from positiveArray and negative ones are chosen to complement allArray.
        """
        from sandbox.recommendation.MaxLocalAUCCython import objectiveApprox
        
        indPtr, colInds = positiveArray
        U = numpy.ascontiguousarray(U)
        V = numpy.ascontiguousarray(V)        
        
        if allArray == None: 
            return objectiveApprox(indPtr, colInds, indPtr, colInds, U,  V, r, gi, gp, gq, self.numRecordAucSamples, self.rho, full=full)         
        else:
            allIndPtr, allColInds = allArray
            return objectiveApprox(indPtr, colInds, allIndPtr, allColInds, U,  V, r, gi, gp, gq, self.numRecordAucSamples, self.rho, full=full)
  
    def __str__(self): 
        outputStr = "MaxLocalAUC: k=" + str(self.k) + " eps=" + str(self.eps) 
        outputStr += " stochastic=" + str(self.stochastic) + " numRowSamples=" + str(self.numRowSamples) 
        outputStr += " numAucSamples=" + str(self.numAucSamples) + " maxIterations=" + str(self.maxIterations) + " initialAlg=" + self.initialAlg
        outputStr += " w=" + str(self.w) + " rho=" + str(self.rho) + " rate=" + str(self.rate) + " alpha=" + str(self.alpha) + " t0=" + str(self.t0) 
        outputStr += " lmbdaU=" + str(self.lmbdaU) + " lmbdaV=" + str(self.lmbdaV) + " sampling=" + str(self.sampling) + " recordStep=" + str(self.recordStep)
        outputStr += " itemExpP=" + str(self.itemExpP) + " itemExpQ=" + str(self.itemExpQ)
        outputStr += super(MaxLocalAUC, self).__str__()
        
        
        return outputStr 

    def modelParamsStr(self): 
        outputStr = " lmbdaU=" + str(self.lmbdaU) + " lmbdaV=" + str(self.lmbdaV) + " k=" + str(self.k) + " rho=" + str(self.rho)
        return outputStr 

    def copy(self): 
        
        maxLocalAuc = MaxLocalAUC(k=self.k, w=self.w, lmbdaU=self.lmbdaU, lmbdaV=self.lmbdaV)
        self.copyParams(maxLocalAuc)
        maxLocalAuc.eps = self.eps 
        maxLocalAuc.stochastic = self.stochastic
        maxLocalAuc.rho = self.rho 
     
        maxLocalAuc.rate = self.rate
        maxLocalAuc.alpha = self.alpha
        maxLocalAuc.t0 = self.t0
        maxLocalAuc.beta = self.beta
        
        maxLocalAuc.recordStep = self.recordStep
        maxLocalAuc.numRowSamples = self.numRowSamples
        maxLocalAuc.numAucSamples = self.numAucSamples
        maxLocalAuc.numRecordAucSamples = self.numRecordAucSamples
        maxLocalAuc.maxIterations = self.maxIterations
        maxLocalAuc.initialAlg = self.initialAlg
        maxLocalAuc.sampling = self.sampling

        maxLocalAuc.itemExpP = self.itemExpP
        maxLocalAuc.itemExpQ = self.itemExpQ
        
        maxLocalAuc.ks = self.ks
        maxLocalAuc.lmbdas = self.lmbdas
        
        return maxLocalAuc
        