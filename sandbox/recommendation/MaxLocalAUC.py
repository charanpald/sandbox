
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
from sandbox.misc.RandomisedSVD import RandomisedSVD

def computeObjective(args): 
    """
    Compute the objective for a particular parameter set. Used to set a learning rate. 
    """
    X, U, V, maxLocalAuc  = args 
    U, V, trainObjs, trainAucs, testObjs, testAucs, precisions, iterations, totalTime = maxLocalAuc.learnModel(X, U=U, V=V, verbose=True)
    obj = trainObjs[-1]

        
    logging.debug("Final objective: " + str(obj) + " with t0=" + str(maxLocalAuc.t0) + " and alpha=" + str(maxLocalAuc.alpha))
    return obj
    
def computeTestAuc(args): 
    trainX, testX, U, V, maxLocalAuc  = args 
    
    U, V, trainObjs, trainAucs, testObjs, testAucs, precisions, iterations, totalTime = maxLocalAuc.learnModel(trainX, U=U, V=V, verbose=True)
    muAuc = numpy.average(testAucs, weights=numpy.flipud(1/numpy.arange(1, len(testAucs)+1, dtype=numpy.float)))
    logging.debug("Weighted local AUC: " + str('%.4f' % muAuc) + " with k=" + str(maxLocalAuc.k) + " lmbda=" + str(maxLocalAuc.lmbda) + " rho=" + str(maxLocalAuc.rho))
        
    return muAuc
    
def computeTestPrecision(args): 
    trainX, testX, U, V, maxLocalAuc = args 
    
    #logging.debug("About to learn")
    U, V, trainObjs, trainAucs, testObjs, testAucs, precisions, iterations, totalTime = maxLocalAuc.learnModel(trainX, verbose=True)
    testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, maxLocalAuc.validationSize, trainX)
    precision = MCEvaluator.precisionAtK(SparseUtils.getOmegaListPtr(testX), testOrderedItems, maxLocalAuc.validationSize)

    logging.debug("Precision@" + str(maxLocalAuc.validationSize) + ": " + str('%.4f' % precision) + " with k=" + str(maxLocalAuc.k) + " lmbda=" + str(maxLocalAuc.lmbda) + " rho=" + str(maxLocalAuc.rho))
        
    return precision
      
class MaxLocalAUC(object): 
    def __init__(self, k, w, alpha=0.05, eps=10**-6, lmbda=0.001, stochastic=False, numProcesses=None): 
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
        self.k = k 
        self.w = w
        self.eps = eps 
        self.stochastic = stochastic

        if numProcesses == None: 
            self.numProcesses = multiprocessing.cpu_count()

        self.chunkSize = 1        
        
        self.rate = "constant"
        self.alpha = alpha #Initial learning rate 
        self.t0 = 0.1 #Convergence speed - larger means we get to 0 faster
        self.beta = 0.75
        
        self.normalise = True
        self.lmbda = lmbda 
        self.rho = 1.00 #Penalise low rank elements 
        
        self.recordStep = 10
        self.numRowSamples = 100
        self.numAucSamples = 10
        self.numRecordAucSamples = 500
        #1 iterations is a complete run over the dataset (i.e. m gradients)
        self.maxIterations = 50
        self.initialAlg = "rand"
        #Possible choices are uniform, top, rank 
        self.sampling = "uniform"
        #The number of items to use to compute precision, sample for probabilities etc.         
        self.z = 5
        
        #Model selection parameters 
        self.folds = 2 
        self.validationSize = 3
        self.validationUsers = 0.1
        self.ks = 2**numpy.arange(3, 8)
        self.lmbdas = numpy.linspace(0.5, 2.0, 7)
        self.rhos = numpy.array([0, 0.1, 0.5, 1.0])
        self.metric = "auc"

        #Learning rate selection 
        self.t0s = 10**-numpy.arange(2, 5, 0.5)
        self.alphas = 2.0**-numpy.arange(-1, 3, 0.5)
    
    def recordResults(self, muU, muV, trainMeasures, testMeasures, sigma, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, c, trainX): 
        r = SparseUtilsCython.computeR(muU, muV, self.w, self.numRecordAucSamples)
        objArr = self.objectiveApprox((indPtr, colInds), muU, muV, r, c, full=True)
        trainMeasures.append([objArr.mean(), MCEvaluator.localAUCApprox((indPtr, colInds), muU, muV, self.w, self.numRecordAucSamples, r)]) 
        
        testMeasuresRow = []
        testMeasuresRow.append(self.objectiveApprox((testIndPtr, testColInds), muU, muV, r, c, allArray=(allIndPtr, allColInds)))
        testMeasuresRow.append(MCEvaluator.localAUCApprox((testIndPtr, testColInds), muU, muV, self.w, self.numRecordAucSamples, r, allArray=(allIndPtr, allColInds)))
        testOrderedItems = MCEvaluatorCython.recommendAtk(muU, muV, self.z, trainX)
        precisionArray, orderedItems = MCEvaluator.precisionAtK((testIndPtr, testColInds), testOrderedItems, self.z, verbose=True)
        testMeasuresRow.append(precisionArray[rowSamples].mean())   
        mrr = MCEvaluatorCython.reciprocalRankAtk(testIndPtr, testColInds, testOrderedItems)
        testMeasuresRow.append(mrr[rowSamples].mean())
        testMeasures.append(testMeasuresRow)
           
        printStr = "iteration " + str(loopInd) + ":"
        printStr += " sigma=" + str('%.4f' % sigma)
        printStr += " obj~" + str('%.4f' % trainMeasures[-1][0]) 
        printStr += " train: LAUC~" + str('%.4f' % trainMeasures[-1][1]) 
        printStr += " obj~" + str('%.4f' % testMeasuresRow[0])
        printStr += " validation: LAUC~" + str('%.4f' % testMeasuresRow[1])
        printStr += " p@" + str(self.z) + "=" + str('%.4f' % testMeasuresRow[2])
        printStr += " mrr@" + str(self.z) + "=" + str('%.4f' % testMeasuresRow[3])
        printStr += " ||U||=" + str('%.3f' % numpy.linalg.norm(muU))
        printStr += " ||V||=" + str('%.3f' %  numpy.linalg.norm(muV))
        
        return printStr
    
    def learnModel(self, X, verbose=False, U=None, V=None): 
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
        bestPrecision = 0 
        bestU = 0 
        bestV = 0
        
        trainMeasures = []
        testMeasures = []        
    
        loopInd = 0
        gradientInd = 0
        
        #Set up order of indices for stochastic methods 
        permutedRowInds = numpy.array(numpy.random.permutation(m), numpy.uint32)
        permutedColInds = numpy.array(numpy.random.permutation(n), numpy.uint32)
        
        startTime = time.time()
        self.wv = 1 - X.sum(1)/float(n)
        
        #A more popular item has a lower weight 
        c = (1/(X.sum(0)+1))**0.5
        c = c/c.mean()
        #print(c)
        #print(numpy.min(c), numpy.max(c))
        #c = numpy.ones(n)
        #print(c)
    
        while loopInd < self.maxIterations:           
            if self.rate == "constant": 
                sigma = self.alpha 
            elif self.rate == "optimal":
                sigma = self.alpha/((1 + self.alpha*self.t0*loopInd**self.beta))
            else: 
                raise ValueError("Invalid rate: " + self.rate)
            
            if loopInd % self.recordStep == 0: 
   
                printStr = self.recordResults(muU, muV, trainMeasures, testMeasures, sigma, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, c, trainX)
                               
                if loopInd != 0: 
                    print("")  
                    
                logging.debug(printStr) 
                                
                if testMeasures[-1][2] >= bestPrecision: 
                    bestPrecision = testMeasures[-1][2]
                    bestU = muU 
                    bestV = muV 
                
            U  = numpy.ascontiguousarray(U)
            
            self.updateUV(indPtr, colInds, U, V, muU, muV, permutedRowInds, permutedColInds, c, gradientInd, sigma)                       
                
            loopInd += 1
            
            if self.stochastic: 
                gradientInd = loopInd*m                
            else: 
                gradientInd = loopInd
            
        #Compute quantities for last U and V 
        totalTime = time.time() - startTime
        printStr = "\nTotal iterations: " + str(loopInd)
        printStr += " time=" + str('%.1f' % totalTime) + " "
        printStr += self.recordResults(muU, muV, trainMeasures, testMeasures, sigma, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, c, trainX)
        logging.debug(printStr)
         
        self.U = bestU 
        self.V = bestV
        self.c = c
         
        if verbose:     
            return self.U, self.V, numpy.array(trainMeasures), numpy.array(testMeasures), loopInd, totalTime
        else: 
            return self.U, self.V
      
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
        maxNorm = numpy.sqrt(numpy.max(numpy.sum(U**2, 1)))
        U = U/maxNorm  
        
        V = numpy.ascontiguousarray(V) 
        #maxNorm = numpy.sqrt(numpy.max(numpy.sum(V**2, 1)))
        #V = V/maxNorm
        
        return U, V
        
    def updateUV(self, indPtr, colInds, U, V, muU, muV, permutedRowInds, permutedColInds, c, ind, sigma): 
        """
        Find the derivative with respect to V or part of it. 
        """
        if not self.stochastic:               
            r = SparseUtilsCython.computeR(U, V, self.w, self.numRecordAucSamples)  
            #r = SparseUtilsCython.computeR2(U, V, self.wv, self.numRecordAucSamples)
            updateU(indPtr, colInds, U, V, r, sigma, self.lmbda, self.rho, self.normalise)
            updateV(indPtr, colInds, U, V, r, sigma, self.lmbda, self.rho, self.normalise)
            
            muU[:] = U[:] 
            muV[:] = V[:]
        else: 
            if self.sampling == "uniform": 
                colIndsCumProbs = self.omegaProbsUniform(indPtr, colInds, muU, muV)
            elif self.sampling == "top": 
                colIndsCumProbs = self.omegaProbsTopZ(indPtr, colInds, muU, muV)
            elif self.sampling == "rank": 
                colIndsCumProbs = self.omegaProbsRank(indPtr, colInds, muU, muV)
            else: 
                raise ValueError("Unknown sampling scheme: " + self.sampling)
            
            updateUVApprox(indPtr, colInds, U, V, muU, muV, colIndsCumProbs, permutedRowInds, permutedColInds, c, ind, sigma, self.numRowSamples, self.numAucSamples, self.w, self.lmbda, self.rho, self.normalise)

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
            ri = numpy.sort(uiVOmegai)[-min(self.z, uiVOmegai.shape[0])]
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
        
    def modelSelect(self, X): 
        """
        Perform model selection on X and return the best parameters. 
        """
        m, n = X.shape
        trainTestXs = Sampling.shuffleSplitRows(X, self.folds, self.validationSize)
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
                        maxLocalAuc.lmbda = lmbda
                        maxLocalAuc.rho = rho 
                    
                        paramList.append((trainX, testX, U.copy(), V.copy(), maxLocalAuc))
            
        logging.debug("Set parameters")
        if self.numProcesses != 1: 
            pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
            if self.metric == "auc": 
                resultsIterator = pool.imap(computeTestAuc, paramList, self.chunkSize)
            elif self.metric == "precision": 
                resultsIterator = pool.imap(computeTestPrecision, paramList, self.chunkSize)
        else: 
            import itertools
            if self.metric == "auc": 
                resultsIterator = itertools.imap(computeTestAuc, paramList)
            elif self.metric == "precision": 
                resultsIterator = itertools.imap(computeTestPrecision, paramList)
        
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
        self.lmbda = self.lmbdas[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[1]]
        self.rho = self.rhos[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[2]]

        logging.debug("Model parameters: k=" + str(self.k) + " lmbda=" + str(self.lmbda) + " rho=" + str(self.rho) +  " max=" + str(numpy.max(meanTestMetrics)))
         
        return meanTestMetrics, stdTestMetrics
  
    def objectiveApprox(self, positiveArray, U, V, r, c, allArray=None, full=False): 
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
            return objectiveApprox(indPtr, colInds, indPtr, colInds, U,  V, r, c, self.numRecordAucSamples, self.rho, full=full)         
        else:
            allIndPtr, allColInds = allArray
            return objectiveApprox(indPtr, colInds, allIndPtr, allColInds, U,  V, r, c, self.numRecordAucSamples, self.rho, full=full)
  
    def __str__(self): 
        outputStr = "MaxLocalAUC: k=" + str(self.k) + " eps=" + str(self.eps) 
        outputStr += " stochastic=" + str(self.stochastic) + " numRowSamples=" + str(self.numRowSamples) 
        outputStr += " numAucSamples=" + str(self.numAucSamples) + " maxIterations=" + str(self.maxIterations) + " initialAlg=" + self.initialAlg
        outputStr += " w=" + str(self.w) + " rho=" + str(self.rho) + " rate=" + str(self.rate) + " alpha=" + str(self.alpha) + " t0=" + str(self.t0) + " folds=" + str(self.folds)
        outputStr += " lmbda=" + str(self.lmbda) +  " numProcesses=" + str(self.numProcesses) + " validationSize=" + str(self.validationSize)
        outputStr += " sampling=" + str(self.sampling) + " z=" + str(self.z) + " recordStep=" + str(self.recordStep)
        
        return outputStr 

    def copy(self): 
        maxLocalAuc = MaxLocalAUC(k=self.k, w=self.w, lmbda=self.lmbda)
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
        maxLocalAuc.z = self.z
        
        maxLocalAuc.ks = self.ks
        maxLocalAuc.lmbdas = self.lmbdas
        maxLocalAuc.folds = self.folds
        maxLocalAuc.validationSize = self.validationSize
        
        return maxLocalAuc
        