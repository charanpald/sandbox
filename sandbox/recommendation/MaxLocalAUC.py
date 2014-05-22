
import numpy 
import logging
import multiprocessing 
import sppy 
import time
import scipy.sparse
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.recommendation.MaxLocalAUCCython import derivativeUi, derivativeVi, updateUVApprox, objectiveApprox, localAUCApprox, updateV, updateU
from sandbox.util.Sampling import Sampling 
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython 
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute 
from sandbox.recommendation.WeightedMf import WeightedMf

def computeObjective(args): 
    """
    Compute the objective for a particular parameter set. Used to set a learning rate. 
    """
    X, omegaList, U, V, maxLocalAuc  = args 
    U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, totalTime = maxLocalAuc.learnModel(X, U=U, V=V, verbose=True)
    obj = trainObjs[-1]
        
    logging.debug("Final objective: " + str(obj) + " with t0=" + str(maxLocalAuc.t0) + " and alpha=" + str(maxLocalAuc.alpha))
    return obj
    
def computeTestAuc(args): 
    trainX, testX, U, V, maxLocalAuc  = args 
    
    logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

    
    U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, totalTime = maxLocalAuc.learnModel(trainX, testX=testX, U=U, V=V, verbose=True)
    muAuc = numpy.average(testAucs, weights=numpy.flipud(1/numpy.arange(1, len(testAucs)+1, dtype=numpy.float)))
    logging.debug("Weighted local AUC: " + str(muAuc) + " with k=" + str(maxLocalAuc.k) + " lmbda=" + str(maxLocalAuc.lmbda) + " C=" + str(maxLocalAuc.C))
        
    return muAuc
    
def computeTestPrecision(args): 
    trainX, testX, U, V, maxLocalAuc = args 
    p = maxLocalAuc.validationSize 
    testOmegaList = SparseUtils.getOmegaList(testX)
    
    #logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
    
    U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, totalTime = maxLocalAuc.learnModel(trainX, testX=testX, U=U, V=V, verbose=True)
    
    testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, p, trainX)
    precision = MCEvaluator.precisionAtK(testX, testOrderedItems, p, omegaList=testOmegaList)
    logging.debug("Precision@" + str(maxLocalAuc.validationSize) + ": " + str(precision) + " with k=" + str(maxLocalAuc.k) + " lmbda=" + str(maxLocalAuc.lmbda) + " C=" + str(maxLocalAuc.C))
        
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
        
        #Optimal rate doesn't seem to work 
        self.rate = "constant"
        self.alpha = alpha #Initial learning rate 
        self.t0 = 0.1 #Convergence speed - larger means we get to 0 faster
        self.beta = 0.75
        
        self.normalise = True
        self.lmbda = lmbda 
        self.C = 0.00 #Penalty on orthogonality constraint ||U^TU - I|| + ||V^TV - I|| 
        
        self.recordStep = 20
        self.numRowSamples = 20
        self.numStepIterations = 100
        self.numAucSamples = 50
        self.numRecordAucSamples = 500
        #1 iterations is a complete run over the dataset (i.e. m gradients)
        self.maxIterations = 50
        self.initialAlg = "rand"
        
        #Model selection parameters 
        self.folds = 2 
        self.validationSize = 3
        self.ks = 2**numpy.arange(3, 8)
        self.lmbdas = 2.0**-numpy.arange(1, 10, 2)
        self.Cs = 2.0**-numpy.arange(0, 10, 2)
        self.metric = "auc"

        #Learning rate selection 
        self.alphas = 2.0**-numpy.arange(0, 9, 1)
        self.t0s = numpy.logspace(-1, -4, 6, base=10)
    
    def learnModel(self, X, verbose=False, U=None, V=None, testX=None): 
        """
        Max local AUC with Frobenius norm penalty on V. Solve with gradient descent. 
        The input is a sparse array. 
        """
        m = X.shape[0]
        n = X.shape[1]
        omegaList = SparseUtils.getOmegaList(X)
        #Not that to compute the test AUC we pick i \in X and j \notin X \cup testX        
        if testX != None: 
            testOmegaList = SparseUtils.getOmegaList(testX)
            allX = X+testX

        if U==None or V==None:
            U, V = self.initUV(X)
        
        lastU = numpy.random.rand(m, self.k)
        lastV = numpy.random.rand(n, self.k)
        lastObj = 0
        obj = 2
        
        xi = numpy.ones(m)*0.5   
        #xi = numpy.zeros(m) 
        
        muU = U.copy() 
        muV = V.copy()
        muXi = xi.copy()
        
        trainObjs = []
        trainAucs = []
        testObjs = []
        testAucs = []
        
        ind = 0
        
        #Convert to a csarray for faster access 
        if scipy.sparse.issparse(X):
            logging.debug("Converting to csarray")
            X2 = sppy.csarray(X, storagetype="row")
            X = X2
        
        #Set up order of indices for stochastic methods 
        rowInds = numpy.array(numpy.random.permutation(m), numpy.uint32)
        colInds = numpy.array(numpy.random.permutation(n), numpy.uint32)
        
        startTime = time.time()
    
        while ind < self.maxIterations*m and abs(obj- lastObj) > self.eps:           
            if self.rate == "constant": 
                sigma = self.alpha 
            elif self.rate == "optimal":
                sigma = self.alpha/((1 + self.alpha*self.t0*ind**self.beta))
            else: 
                raise ValueError("Invalid rate: " + self.rate)
            
            if ind % self.recordStep == 0: 
                r = SparseUtilsCython.computeR(muU, muV, self.w, self.numRecordAucSamples)
                trainObjs.append(objectiveApprox(X, muU, muV, omegaList, self.numRecordAucSamples, muXi, self.lmbda, self.C))
                trainAucs.append(localAUCApprox(X, muU, muV, omegaList, self.numRecordAucSamples, r))
                
                if testX != None:
                    testObjs.append(objectiveApprox(allX, muU, muV, testOmegaList, self.numRecordAucSamples, muXi, self.lmbda, self.C))
                    testAucs.append(localAUCApprox(allX, muU, muV, testOmegaList, self.numRecordAucSamples, r))
                    p = 5
                    testOrderedItems = MCEvaluatorCython.recommendAtk(muU, muV, p, X)
                    precision = MCEvaluator.precisionAtK(testX, testOrderedItems, p, omegaList=testOmegaList)                    
                    
                printStr = "Iteration: " + str(ind)
                printStr += " LAUC~" + str('%.4f' % trainAucs[-1]) 
                printStr += " obj~" + str('%.4f' % trainObjs[-1]) 
                if testX != None:
                    printStr += " test LAUC~" + str('%.4f' % testAucs[-1])
                    printStr += " test obj~" + str('%.4f' % testObjs[-1])
                    printStr += " test precision=" + str('%.3f' % precision) 
                printStr += " sigma=" + str('%.4f' % sigma)
                printStr += " normU=" + str('%.3f' % numpy.linalg.norm(U))
                printStr += " normV=" + str('%.3f' %  numpy.linalg.norm(V))
                
                logging.debug(printStr)
                
                
                lastObj = obj
                obj = numpy.average(trainObjs, weights=numpy.flipud(1/numpy.arange(1, len(trainObjs)+1, dtype=numpy.float)))
            
            lastU = U.copy() 
            lastV = V.copy()
            
            U  = numpy.ascontiguousarray(U)
            
            self.updateUV(X, U, V, lastU, lastV, muU, muV, xi, muXi, rowInds, colInds, ind, omegaList, sigma)                       
                            
            if self.stochastic: 
                ind += self.numStepIterations
            else: 
                ind += 1
            
        #Compute quantities for last U and V 
        r = SparseUtilsCython.computeR(muU, muV, self.w, self.numRecordAucSamples)
        trainObjs.append(objectiveApprox(X, muU, muV, omegaList, self.numRecordAucSamples, muXi, self.lmbda, self.C))
        trainAucs.append(localAUCApprox(X, muU, muV, omegaList, self.numRecordAucSamples, r))
        
        if testX != None:
            testObjs.append(objectiveApprox(allX, muU, muV, testOmegaList, self.numRecordAucSamples, muXi, self.lmbda, self.C))
            testAucs.append(localAUCApprox(allX, muU, muV, testOmegaList, self.numRecordAucSamples, r))            
            
        totalTime = time.time() - startTime
        printStr = "Total iterations: " + str(ind)
        printStr += " time=" + str('%.1f' % totalTime) 
        printStr += " LAUC~" + str('%.4f' % trainAucs[-1]) 
        printStr += " obj~" + str('%.4f' % trainObjs[-1]) 
        if testX != None:
            printStr += " test LAUC~" + str('%.4f' % testAucs[-1])
            printStr += " test obj~" + str('%.4f' % testObjs[-1])
        printStr += " sigma=" + str('%.4f' % sigma)
        printStr += " normU=" + str('%.3f' % numpy.linalg.norm(U))
        printStr += " normV=" + str('%.3f' %  numpy.linalg.norm(V))
        logging.debug(printStr)
                  
        self.U = muU 
        self.V = muV                  
                  
        if verbose:     
            return muU, muV, numpy.array(trainObjs), numpy.array(trainAucs), numpy.array(testObjs), numpy.array(testAucs), ind, totalTime
        else: 
            return muU, muV
      
    def predict(self, maxItems): 
        return MCEvaluator.recommendAtk(self.U, self.V, maxItems)
          
    def initUV(self, X): 
        m = X.shape[0]
        n = X.shape[1]        
        
        if self.initialAlg == "rand": 
            U = numpy.random.randn(m, self.k)
            V = numpy.random.randn(n, self.k)
        elif self.initialAlg == "svd":
            logging.debug("Initialising with SVD")
            try: 
                U, s, V = SparseUtils.svdPropack(X, self.k, kmax=numpy.min([self.k*15, m-1, n-1]))
            except ImportError: 
                U, s, V = SparseUtils.svdArpack(X, self.k)
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
        
    def updateUV(self, X, U, V, lastU, lastV, muU, muV, xi, muXi, rowInds, colInds, ind, omegaList, sigma): 
        """
        Find the derivative with respect to V or part of it. 
        """
        if not self.stochastic:                 
            r = SparseUtilsCython.computeR(U, V, self.w, self.numAucSamples)
            updateU(X, U, V, omegaList, sigma, r, self.nu)
            updateV(X, U, V, omegaList, sigma, r, self.nu, self.lmbda)
        else: 
            omegaProbabilitiesList = self.omegaProbabilities2(muU, muV, omegaList)
            updateUVApprox(X, U, V, muU, muV, xi, muXi, omegaList, omegaProbabilitiesList, rowInds, colInds, ind, sigma, self.numStepIterations, self.numRowSamples, self.numAucSamples, self.w, self.lmbda, self.C, self.normalise)
       
    #@profile
    def derivativeUi(self, X, U, V, omegaList, i, r): 
        """
        delta phi/delta u_i
        """
        return derivativeUi(X, U, V, omegaList, i, r, self.lmbda, self.C, self.normalise)
        
    def derivativeVi(self, X, U, V, omegaList, i, r): 
        """
        delta phi/delta v_i
        """
        return derivativeVi(X, U, V, omegaList, i, r, self.lmbda, self.C, self.normalise)           

    def omegaProbabilities(self, U, V, omegaList): 
        omegaProbabilitiesList = []
        
        for i, omegai in enumerate(omegaList):
            uiVOmegai = U[i, :].T.dot(V[omegai, :].T)
            uiVOmegai = numpy.exp(uiVOmegai)
            omegaProbabilitiesList.append(uiVOmegai/uiVOmegai.sum()) 
            
        return omegaProbabilitiesList
            
    def omegaProbabilities2(self, U, V, omegaList): 
        omegaProbabilitiesList = []
        
        for i, omegai in enumerate(omegaList):
            uiVOmegai = U[i, :].T.dot(V[omegai, :].T)
            inds = numpy.argsort(numpy.argsort((uiVOmegai)))+1
            omegaProbabilitiesList.append(inds/float(inds.sum())) 
            
        return omegaProbabilitiesList


    def learningRateSelect(self, X): 
        """
        Let's set the initial learning rate. 
        """        
        m, n = X.shape
        omegaList = SparseUtils.getOmegaList(X)
        objectives = numpy.zeros((self.t0s.shape[0], self.alphas.shape[0]))
        
        paramList = []   
        logging.debug("t0s=" + str(self.t0s))
        logging.debug("alphas=" + str(self.alphas))
        
        if self.initialAlg == "rand": 
            numInitalUVs = self.folds
        else: 
            numInitalUVs = 1
            
        for k in range(numInitalUVs):
            U, V = self.initUV(X)
                        
            for i, t0 in enumerate(self.t0s): 
                for j, alpha in enumerate(self.alphas): 
                    maxLocalAuc = self.copy()
                    maxLocalAuc.t0 = t0
                    maxLocalAuc.alpha = alpha 
                    paramList.append((X, omegaList, U, V, maxLocalAuc))
                    
        pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
        resultsIterator = pool.imap(computeObjective, paramList, self.chunkSize)
        #import itertools
        #resultsIterator = itertools.imap(computeObjective, paramList)
        
        for k in range(numInitalUVs):
            for i, t0 in enumerate(self.t0s): 
                for j, alpha in enumerate(self.alphas):  
                    objectives[i, j] += resultsIterator.next()
            
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
        testAucs = numpy.zeros((self.ks.shape[0], self.lmbdas.shape[0], self.Cs.shape[0], len(trainTestXs)))
        
        logging.debug("Performing model selection with test leave out per row of " + str(self.validationSize))
        paramList = []        
        
        for i, k in enumerate(self.ks): 
            self.k = k
            
            for icv, (trainX, testX) in enumerate(trainTestXs):
                U, V = self.initUV(trainX)
                for j, lmbda in enumerate(self.lmbdas): 
                    for s, C in enumerate(self.Cs):
                        maxLocalAuc = self.copy()
                        maxLocalAuc.k = k    
                        maxLocalAuc.lmbda = lmbda
                        maxLocalAuc.C = C 
                    
                        paramList.append((trainX, testX, U, V, maxLocalAuc))
            
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
                    for s, C in enumerate(self.Cs):
                        testAucs[i, j, s, icv] = resultsIterator.next()
        
        if self.numProcesses != 1: 
            pool.terminate()
        
        meanTestMetrics = numpy.mean(testAucs, 3)
        stdTestMetrics = numpy.std(testAucs, 3)
        
        logging.debug("ks=" + str(self.ks)) 
        logging.debug("lmbdas=" + str(self.lmbdas)) 
        logging.debug("Cs=" + str(self.Cs)) 
        logging.debug("Mean metrics =" + str(meanTestMetrics))
        
        self.k = self.ks[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[0]]
        self.lmbda = self.lmbdas[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[1]]
        self.C = self.Cs[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[2]]

        logging.debug("Model parameters: k=" + str(self.k) + " lmbda=" + str(self.lmbda) + " C=" + str(self.C))
         
        return meanTestMetrics, stdTestMetrics
    
    def __str__(self): 
        outputStr = "MaxLocalAUC: k=" + str(self.k) + " eps=" + str(self.eps) 
        outputStr += " stochastic=" + str(self.stochastic) + " numRowSamples=" + str(self.numRowSamples) + " numStepIterations=" + str(self.numStepIterations)
        outputStr += " numAucSamples=" + str(self.numAucSamples) + " maxIterations=" + str(self.maxIterations) + " initialAlg=" + self.initialAlg
        outputStr += " w=" + str(self.w) + " rate=" + str(self.rate) + " alpha=" + str(self.alpha) + " t0=" + str(self.t0) + " folds=" + str(self.folds)
        outputStr += " lmbda=" + str(self.lmbda) + " C=" + str(self.C) + " numProcesses=" + str(self.numProcesses) + " validationSize=" + str(self.validationSize)
        
        return outputStr 

    def copy(self): 
        maxLocalAuc = MaxLocalAUC(k=self.k, w=self.w, lmbda=self.lmbda)
        maxLocalAuc.eps = self.eps 
        maxLocalAuc.stochastic = self.stochastic
        maxLocalAuc.C = self.C 
     
        maxLocalAuc.rate = self.rate
        maxLocalAuc.alpha = self.alpha
        maxLocalAuc.t0 = self.t0
        maxLocalAuc.beta = self.beta
        
        maxLocalAuc.recordStep = self.recordStep
        maxLocalAuc.numRowSamples = self.numRowSamples
        maxLocalAuc.numStepIterations = self.numStepIterations
        maxLocalAuc.numAucSamples = self.numAucSamples
        maxLocalAuc.numRecordAucSamples = self.numRecordAucSamples
        maxLocalAuc.maxIterations = self.maxIterations
        maxLocalAuc.initialAlg = self.initialAlg
        
        maxLocalAuc.ks = self.ks
        maxLocalAuc.lmbdas = self.lmbdas
        maxLocalAuc.folds = self.folds
        maxLocalAuc.validationSize = self.validationSize
        
        return maxLocalAuc
        