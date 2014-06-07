
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
    U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, totalTime = maxLocalAuc.learnModel(X, U=U, V=V, verbose=True)
    obj = trainObjs[-1]

        
    logging.debug("Final objective: " + str(obj) + " with t0=" + str(maxLocalAuc.t0) + " and alpha=" + str(maxLocalAuc.alpha))
    return obj
    
def computeTestAuc(args): 
    trainX, testX, U, V, maxLocalAuc  = args 
    
    U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, totalTime = maxLocalAuc.learnModel(trainX, U=U, V=V, verbose=True)
    muAuc = numpy.average(testAucs, weights=numpy.flipud(1/numpy.arange(1, len(testAucs)+1, dtype=numpy.float)))
    logging.debug("Weighted local AUC: " + str('%.4f' % muAuc) + " with k=" + str(maxLocalAuc.k) + " lmbda=" + str(maxLocalAuc.lmbda) + " rho=" + str(maxLocalAuc.rho))
        
    return muAuc
    
def computeTestPrecision(args): 
    trainX, testX, U, V, maxLocalAuc = args 
    
    #logging.debug("About to learn")
    U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, totalTime = maxLocalAuc.learnModel(trainX, verbose=True)
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
        
        self.recordStep = 5
        self.numRowSamples = 100
        self.numAucSamples = 10
        self.numRecordAucSamples = 500
        #1 iterations is a complete run over the dataset (i.e. m gradients)
        self.maxIterations = 50
        self.initialAlg = "rand"
        #Possible choices are uniform, top, rank 
        self.sampling = "uniform"
        #The number of items to use to compute precision, sample for probabilities etc.         
        self.z = 1
        
        #Model selection parameters 
        self.folds = 2 
        self.validationSize = 3
        self.validationUsers = 0.2
        self.ks = 2**numpy.arange(3, 8)
        self.lmbdas = 2.0**-numpy.arange(-1, 6)
        self.metric = "auc"

        #Learning rate selection 
        self.t0s = 10**-numpy.arange(2, 5, 0.5)
        self.alphas = 2.0**-numpy.arange(-1, 3, 0.5)
    
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
        numValidationUsers = int(X.shape[0]*self.validationUsers)
        trainX, testX, rowSamples = Sampling.shuffleSplitRows(X, 1, self.validationSize, numRows=numValidationUsers)[0]  

        m = trainX.shape[0]
        n = trainX.shape[1]
        indPtr, colInds = SparseUtils.getOmegaListPtr(trainX)
        
        #Not that to compute the test AUC we pick i \in X and j \notin X \cup testX        
        testIndPtr, testColInds = SparseUtils.getOmegaListPtr(testX)
        allIndPtr, allColInds = SparseUtils.getOmegaListPtr(X)

        if U==None or V==None:
            U, V = self.initUV(trainX)
        
        lastU = numpy.random.rand(m, self.k)
        lastV = numpy.random.rand(n, self.k)
        lastObj = 0
        obj = 2
        sigma = self.alpha
        
        muU = U.copy() 
        muV = V.copy()
        
        #Store best results 
        bestPrecision = 0 
        bestU = 0 
        bestV = 0
        
        trainObjs = []
        trainAucs = []
        testObjs = []
        testAucs = []
        precisions = []
        
        loopInd = 0
        gradientInd = 0
        
        #Set up order of indices for stochastic methods 
        permutedRowInds = numpy.array(numpy.random.permutation(m), numpy.uint32)
        permutedColInds = numpy.array(numpy.random.permutation(n), numpy.uint32)
        
        startTime = time.time()
        self.wv = 1 - X.sum(1)/float(n)
    
        while loopInd < self.maxIterations and abs(obj- lastObj) > self.eps:           
            if self.rate == "constant": 
                sigma = self.alpha 
            elif self.rate == "optimal":
                sigma = self.alpha/((1 + self.alpha*self.t0*loopInd**self.beta))
            else: 
                raise ValueError("Invalid rate: " + self.rate)
            
            if loopInd % self.recordStep == 0: 
                r = SparseUtilsCython.computeR(muU, muV, self.w, self.numRecordAucSamples)
                objArr = self.objectiveApprox((indPtr, colInds), muU, muV, r, full=True)
                #userProbs = numpy.array(objArr > objArr.mean(), numpy.float)
                #userProbs /= userProbs.sum()
                #print(userProbs)
                #permutedRowInds = numpy.random.choice(numpy.arange(m, dtype=numpy.uint32), size=m, p=userProbs)
                trainObjs.append(objArr.mean())
                trainAucs.append(MCEvaluator.localAUCApprox((indPtr, colInds), muU, muV, self.w, self.numRecordAucSamples, r))
                testObjs.append(self.objectiveApprox((testIndPtr, testColInds), muU, muV, r, allArray=(allIndPtr, allColInds)))
                testAucs.append(MCEvaluator.localAUCApprox((testIndPtr, testColInds), muU, muV, self.w, self.numRecordAucSamples, r, allArray=(allIndPtr, allColInds)))
                testOrderedItems = MCEvaluatorCython.recommendAtk(muU, muV, self.validationSize, trainX)
                precisionArray, orderedItems = MCEvaluator.precisionAtK((testIndPtr, testColInds), testOrderedItems, self.validationSize, verbose=True)
                precisions.append(precisionArray[rowSamples].mean())   
                   
                printStr = "Iteration " + str(loopInd) + ":"
                printStr += " sigma=" + str('%.4f' % sigma)
                printStr += " train: LAUC~" + str('%.4f' % trainAucs[-1]) 
                printStr += " obj~" + str('%.4f' % trainObjs[-1]) 
                printStr += " validation: LAUC~" + str('%.4f' % testAucs[-1])
                printStr += " obj~" + str('%.4f' % testObjs[-1])
                printStr += " p@" + str(self.validationSize) + "=" + str('%.4f' % precisions[-1])
                printStr += " ||U||=" + str('%.3f' % numpy.linalg.norm(U))
                printStr += " ||V||=" + str('%.3f' %  numpy.linalg.norm(V))
                logging.debug(printStr)
                
                lastObj = obj
                obj = numpy.average(trainObjs, weights=numpy.flipud(1/numpy.arange(1, len(trainObjs)+1, dtype=numpy.float)))
                
                if precisions[-1] > bestPrecision: 
                    bestPrecision = precisions[-1]
                    bestU = muU 
                    bestV = muV 
                
            lastU = U.copy() 
            lastV = V.copy()
            
            U  = numpy.ascontiguousarray(U)
            
            self.updateUV(indPtr, colInds, U, V, lastU, lastV, muU, muV, permutedRowInds, permutedColInds, gradientInd, sigma)                       
                
            loopInd += 1
            
            if self.stochastic: 
                gradientInd = loopInd*m                
            else: 
                gradientInd = loopInd
            
        #Compute quantities for last U and V 
        r = SparseUtilsCython.computeR(muU, muV, self.w, self.numRecordAucSamples)
        trainObjs.append(self.objectiveApprox((indPtr, colInds), muU, muV, r))
        trainAucs.append(MCEvaluator.localAUCApprox((indPtr, colInds), muU, muV, self.w, self.numRecordAucSamples, r))
        testObjs.append(self.objectiveApprox((testIndPtr, testColInds), muU, muV, r, allArray=(allIndPtr, allColInds)))
        testAucs.append(MCEvaluator.localAUCApprox((testIndPtr, testColInds), muU, muV, self.w, self.numRecordAucSamples, r, allArray=(allIndPtr, allColInds)))          
            
        totalTime = time.time() - startTime
        printStr = "Total iterations: " + str(loopInd)
        printStr += " time=" + str('%.1f' % totalTime) 
        printStr += " sigma=" + str('%.4f' % sigma)
        printStr += " train: LAUC~" + str('%.4f' % trainAucs[-1]) 
        printStr += " obj~" + str('%.4f' % trainObjs[-1]) 
        printStr += " test: LAUC~" + str('%.4f' % testAucs[-1])
        printStr += " obj~" + str('%.4f' % testObjs[-1])
        printStr += " ||U||=" + str('%.3f' % numpy.linalg.norm(U))
        printStr += " ||V||=" + str('%.3f' %  numpy.linalg.norm(V))
        logging.debug(printStr)
         
        self.U = bestU 
        self.V = bestV
         
        if verbose:     
            return bestU, bestV, numpy.array(trainObjs), numpy.array(trainAucs), numpy.array(testObjs), numpy.array(testAucs), loopInd, totalTime
        else: 
            return bestU, bestV
      
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
        
    def updateUV(self, indPtr, colInds, U, V, lastU, lastV, muU, muV, permutedRowInds, permutedColInds, ind, sigma): 
        """
        Find the derivative with respect to V or part of it. 
        """
        if not self.stochastic:               
            r = SparseUtilsCython.computeR(U, V, self.w, self.numRecordAucSamples)  
            r = SparseUtilsCython.computeR2(U, V, self.wv, self.numRecordAucSamples)
            updateU(indPtr, colInds, U, V, r, sigma, self.lmbda, self.rho, self.normalise)
            updateV(indPtr, colInds, U, V, r, sigma, self.lmbda, self.rho, self.normalise)
        else: 
            if self.sampling == "uniform": 
                colIndsCumProbs = self.omegaProbsUniform(indPtr, colInds, muU, muV)
            elif self.sampling == "top": 
                colIndsCumProbs = self.omegaProbsTopZ(indPtr, colInds, muU, muV)
            elif self.sampling == "rank": 
                colIndsCumProbs = self.omegaProbsRank(indPtr, colInds, muU, muV)
            else: 
                raise ValueError("Unknown sampling scheme: " + self.sampling)
            
            updateUVApprox(indPtr, colInds, U, V, muU, muV, colIndsCumProbs, permutedRowInds, permutedColInds, ind, sigma, self.numRowSamples, self.numAucSamples, self.w, self.lmbda, self.rho, self.normalise)

    def derivativeUi(self, indPtr, colInds, U, V, r, i): 
        """
        delta phi/delta u_i
        """
        return derivativeUi(indPtr, colInds, U, V, r, i, self.lmbda, self.rho, self.normalise)
        
    def derivativeVi(self, X, U, V, omegaList, i, r): 
        """
        delta phi/delta v_i
        """
        return derivativeVi(X, U, V, omegaList, i, r, self.lmbda, self.rho, self.normalise)           

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
        testAucs = numpy.zeros((self.ks.shape[0], self.lmbdas.shape[0], len(trainTestXs)))
        
        logging.debug("Performing model selection with test leave out per row of " + str(self.validationSize))
        paramList = []        
        
        for i, k in enumerate(self.ks): 
            self.k = k
            
            for icv, (trainX, testX) in enumerate(trainTestXs):
                U, V = self.initUV(trainX)
                for j, lmbda in enumerate(self.lmbdas): 
                    maxLocalAuc = self.copy()
                    maxLocalAuc.k = k    
                    maxLocalAuc.lmbda = lmbda
                
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
                    testAucs[i, j, icv] = resultsIterator.next()
        
        if self.numProcesses != 1: 
            pool.terminate()
        
        meanTestMetrics = numpy.mean(testAucs, 2)
        stdTestMetrics = numpy.std(testAucs, 2)
        
        logging.debug("ks=" + str(self.ks)) 
        logging.debug("lmbdas=" + str(self.lmbdas)) 
        logging.debug("Mean metrics =" + str(meanTestMetrics))
        
        self.k = self.ks[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[0]]
        self.lmbda = self.lmbdas[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[1]]

        logging.debug("Model parameters: k=" + str(self.k) + " lmbda=" + str(self.lmbda) + " max=" + str(numpy.max(meanTestMetrics)))
         
        return meanTestMetrics, stdTestMetrics
  
    def objectiveApprox(self, positiveArray, U, V, r, allArray=None, full=False): 
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
            return objectiveApprox(indPtr, colInds, indPtr, colInds, U,  V, r, self.numRecordAucSamples, self.lmbda, self.rho, full=full)         
        else:
            allIndPtr, allColInds = allArray
            return objectiveApprox(indPtr, colInds, allIndPtr, allColInds, U,  V, r, self.numRecordAucSamples, self.lmbda, self.rho, full=full)
  
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
        