
import numpy 
import logging
import multiprocessing 
import sppy 
import time
import scipy.sparse
from math import exp
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.recommendation.MaxLocalAUCCython import derivativeUi, derivativeVi, updateUVApprox, objectiveApprox, localAUCApprox, updateV, updateU
from sandbox.util.Sampling import Sampling 
from sandbox.util.Util import Util 
from sandbox.data.Standardiser import Standardiser 
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute 


def computeObjective(args): 
    """
    Compute the objective for a particular parameter set. Used to set a learning rate. 
    """
    X, omegaList, U, V, maxLocalAuc  = args 
    U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, totalTime = maxLocalAuc.learnModel(X, U=U, V=V, verbose=True)
    
    muObj = numpy.average(trainObjs, weights=numpy.flipud(1/numpy.arange(1, len(trainObjs)+1)))
    muAuc = -numpy.average(trainAucs, weights=numpy.flipud(1/numpy.arange(1, len(trainAucs)+1)))
    
    #logging.debug("Weighted objective: " + str(muObj) + " with t0=" + str(maxLocalAuc.t0) + " and alpha=" + str(maxLocalAuc.alpha))
    logging.debug("Weighted AUC: " + str(muAuc) + " with t0=" + str(maxLocalAuc.t0) + " and alpha=" + str(maxLocalAuc.alpha))
    return muAuc
    
def computeTestAucs(args): 
    trainX, testX, U, V, maxLocalAuc  = args 
    
    inputU = U.copy() 
    inputV = V.copy()
    testAucScores = numpy.zeros(maxLocalAuc.lmbdas.shape[0])
    logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
    
    for i, lmbda in enumerate(maxLocalAuc.lmbdas):         
        maxLocalAuc.lmbda = lmbda 

        #Don't use warm restarts for the SVD
        if maxLocalAuc.initialAlg == "svd" or maxLocalAuc.initialAlg == "softimpute": 
            U = inputU.copy()
            V = inputV.copy()

        U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, totalTime = maxLocalAuc.learnModel(trainX, testX=testX, U=U, V=V, verbose=True)
            
        muAuc = numpy.average(testAucs, weights=numpy.flipud(1/numpy.arange(1, len(testAucs)+1)))
        testAucScores[i] = muAuc
        
        logging.debug("Weighted local AUC: " + str(muAuc) + " with k=" + str(maxLocalAuc.k) + " lmbda=" + str(maxLocalAuc.lmbda))
        
    return testAucScores
      
class MaxLocalAUC(object): 
    def __init__(self, k, w, alpha=0.05, eps=0.01, lmbda=0.001, stochastic=False, numProcesses=None): 
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
        
        self.nu = 20.0 
        self.lmbda = lmbda 
        self.rho = 0.00 #Penalty on orthogonality constraint ||U^TU - I|| + ||V^TV - I|| 
        
        self.recordStep = 20
        self.numRowSamples = 20
        self.numStepIterations = 100
        self.numAucSamples = 50
        self.numRecordAucSamples = 500
        self.maxIterations = 1000
        self.initialAlg = "rand"
        
        #Model selection parameters 
        self.folds = 5 
        self.testSize = 3
        self.ks = 2**numpy.arange(3, 8)
        self.lmbdas = 2.0**-numpy.arange(1, 10, 2)

        #Learning rate selection 
        self.alphas = 2.0**-numpy.arange(0, 5, 1)
        self.t0s = numpy.logspace(-2, -5, 6, base=10)
    
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
            testX = X+testX

        if U==None or V==None:
            U, V = self.initUV(X)
        
        lastU = numpy.random.rand(m, self.k)
        lastV = numpy.random.rand(n, self.k)
        lastMuObj = 0
        muObj = -1
        
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
    
        while (abs(muObj - lastMuObj) > self.eps) and ind < self.maxIterations:             
            if self.rate == "constant": 
                pass
            elif self.rate == "optimal":
                sigma = self.alpha/((1 + self.alpha*self.t0*ind))
            else: 
                raise ValueError("Invalid rate: " + self.rate)
            
            if ind % self.recordStep == 0: 
                r = SparseUtilsCython.computeR(U, V, self.w, self.numRecordAucSamples)
                trainObjs.append(objectiveApprox(X, U, V, omegaList, self.numRecordAucSamples, r, self.lmbda))
                trainAucs.append(localAUCApprox(X, U, V, omegaList, self.numRecordAucSamples, r))
                
                if testX != None:
                    testObjs.append(objectiveApprox(testX, U, V, testOmegaList, self.numRecordAucSamples, r, self.lmbda))
                    testAucs.append(localAUCApprox(testX, U, V, testOmegaList, self.numRecordAucSamples, r))
                    
                printStr = "Iteration: " + str(ind)
                printStr += " LAUC~" + str(trainAucs[-1]) + " obj~" + str(trainObjs[-1])
                if testX != None:
                    printStr += " test LAUC~" + str(testAucs[-1]) + " obj~" + str(testObjs[-1])    
                printStr += " sigma=" + str(sigma)
                #printStr += " normV=" + str(numpy.linalg.norm(V))
                logging.debug(printStr)

            lastMuObj = muObj
            muObj = numpy.average(trainObjs, weights=numpy.flipud(1/numpy.arange(1, len(trainObjs)+1)))
            
            lastU = U.copy() 
            lastV = V.copy()
            
            U  = numpy.ascontiguousarray(U)
            
            self.updateUV(X, U, V, lastU, lastV, rowInds, colInds, ind, omegaList, sigma)                          
                            
            if self.stochastic: 
                ind += self.numStepIterations
            else: 
                ind += 1
            
        totalTime = time.time() - startTime
        logging.debug("normU=" + str(numpy.linalg.norm(U)) + " normV=" + str(numpy.linalg.norm(V)))
        logging.debug("abs(muObj - lastMuObj)=" + str(abs(muObj - lastMuObj)))
        logging.debug("Total time taken " + str(totalTime))
        logging.debug("Number of iterations: " + str(ind))
        printStr = "Final train local AUC=" + str(trainAucs[-1])
        if testX != None:
            printStr += " test local AUC=" + str(testAucs[-1])
        logging.debug(printStr)
                  
        self.U = U 
        self.V = V                  
                  
        if verbose:     
            return U, V, numpy.array(trainObjs), numpy.array(trainAucs), numpy.array(testObjs), numpy.array(testAucs), ind, totalTime
        else: 
            return U, V
      
    def predict(self, maxItems): 
        return MCEvaluator.recommendAtk(self.U, self.V, maxItems)
          
    def initUV(self, X): 
        m = X.shape[0]
        n = X.shape[1]        
        
        if self.initialAlg == "rand": 
            U = numpy.random.randn(m, self.k)
            V = numpy.random.randn(n, self.k)
            
            #U, R = numpy.linalg.qr(U)
            #V, R = numpy.linalg.qr(V)
        elif self.initialAlg == "svd":
            logging.debug("Initialising with SVD")
            try: 
                U, s, V = SparseUtils.svdPropack(X, self.k, kmax=numpy.min([self.k*15, m-1, n-1]))
            except ImportError: 
                U, s, V = SparseUtils.svdArpack(X, self.k)
            U = numpy.ascontiguousarray(U)
            V = numpy.ascontiguousarray(V)
        elif self.initialAlg == "softimpute": 
            trainIterator = iter([X.toScipyCsc()])
            rho = 0.01
            learner = IterativeSoftImpute(rho, k=self.k, svdAlg="propack", postProcess=True)
            ZList = learner.learnModel(trainIterator)    
            U, s, V = ZList.next()
            U = U*s
            U = numpy.ascontiguousarray(U)
            V = numpy.ascontiguousarray(V)
        else:
            raise ValueError("Unknown initialisation: " + str(self.initialAlg))  
            
        U = Standardiser().normaliseArray(U.T).T    
        V = Standardiser().normaliseArray(V.T).T 
        
        return U, V
        
    def updateUV(self, X, U, V, lastU, lastV, rowInds, colInds, ind, omegaList, sigma): 
        """
        Find the derivative with respect to V or part of it. 
        """
        if not self.stochastic:                 
            r = SparseUtilsCython.computeR(U, V, self.w, self.numAucSamples)
            updateU(X, U, V, omegaList, sigma, r, self.nu)
            updateV(X, U, V, omegaList, sigma, r, self.nu, self.lmbda)
        else: 
            updateUVApprox(X, U, V, omegaList, rowInds, colInds, ind, sigma, self.numStepIterations, self.numRowSamples, self.numAucSamples, self.w, self.nu, self.lmbda, self.rho)
       
    #@profile
    def derivativeUi(self, X, U, V, omegaList, i, r): 
        """
        delta phi/delta u_i
        """
        return derivativeUi(X, U, V, omegaList, i, r, self.nu)
        
    def derivativeVi(self, X, U, V, omegaList, i, r): 
        """
        delta phi/delta v_i
        """
        return derivativeVi(X, U, V, omegaList, i, r, self.nu, self.lmbda)           

    #@profile
    def objective(self, X, U, V, omegaList, r):         
        obj = 0 
        m = X.shape[0]
        
        allInds = numpy.arange(X.shape[1])        
        
        for i in range(X.shape[0]): 
            omegai = omegaList[i]
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            ui = U[i, :]       
            uiV = ui.dot(V.T)
            ri = r[i]
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0                
                
                for p in omegai: 
                    uivp = uiV[p]
                    kappa = numpy.exp(-uivp+ri)
                    onePlusKappa = 1+kappa
                    
                    for q in omegaBari: 
                        uivq = uiV[q]
                        gamma = exp(-uivp+uivq)

                        partialAuc += 1/((1+gamma) * onePlusKappa)
                            
                obj += partialAuc/float(omegai.shape[0] * omegaBari.shape[0])
        
        obj /= m       
        obj = - obj
        
        return obj 

    #@profile
    def objectiveApprox(self, X, U, V, omegaList):         
        obj = 0 
        m = X.shape[0]
        
        allInds = numpy.arange(X.shape[1])        
        
        for i in range(X.shape[0]): 
            omegai = omegaList[i]
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            ui = U[i, :]       
            uiV = ui.dot(V.T)
            ri = self.r[i]
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0                
                
                indsP = numpy.random.randint(0, omegai.shape[0], self.numAucSamples)  
                indsQ = numpy.random.randint(0, omegaBari.shape[0], self.numAucSamples)
                
                for j in range(self.numAucSamples):                    
                    p = omegai[indsP[j]] 
                    q = omegaBari[indsQ[j]]                  
                
                    uivp = uiV[p]
                    kappa = exp(-uivp+ri)
                    
                    uivq = uiV[q]
                    gamma = exp(-uivp+uivq)

                    partialAuc += 1/((1+gamma) * 1+kappa)
                            
                obj += partialAuc/float(self.numAucSamples)
        
        obj /= m       
        obj = - obj
        
        return obj 
        
    def learningRateSelect(self, X): 
        """
        Let's set the initial learning rate. 
        """        
        m, n = X.shape
        omegaList = SparseUtils.getOmegaList(X)
        objectives = numpy.zeros((self.t0s.shape[0], self.alphas.shape[0]))
        
        paramList = []   
        
        if self.initialAlg != "svd" and self.initialAlg != "softimpute": 
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
        objectives /= numInitalUVs   
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
        #cvInds = Sampling.randCrossValidation(self.folds, X.nnz)
        trainTestXs = Sampling.shuffleSplitRows(X, self.folds, self.testSize)
        testAucs = numpy.zeros((self.ks.shape[0], self.lmbdas.shape[0], len(trainTestXs)))
        
        logging.debug("Performing model selection")
        paramList = []        
        
        for i, k in enumerate(self.ks): 
            self.k = k
            U, V = self.initUV(X)
            
            for icv, (trainX, testX) in enumerate(trainTestXs):
                maxLocalAuc = self.copy()
                maxLocalAuc.k = k                
            
                paramList.append((trainX, testX, U, V, maxLocalAuc))
            
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
    
    def __str__(self): 
        outputStr = "MaxLocalAUC: k=" + str(self.k) + " eps=" + str(self.eps) 
        outputStr += " stochastic=" + str(self.stochastic) + " numRowSamples=" + str(self.numRowSamples) + " numStepIterations=" + str(self.numStepIterations)
        outputStr += " numAucSamples=" + str(self.numAucSamples) + " maxIterations=" + str(self.maxIterations) + " initialAlg=" + self.initialAlg
        outputStr += " w=" + str(self.w) + " rate=" + str(self.rate) + " alpha=" + str(self.alpha) + " t0=" + str(self.t0) + " folds=" + str(self.folds)
        outputStr += " nu=" + str(self.nu) + " lmbda=" + str(self.lmbda) + " rho=" + str(self.rho) + " numProcesses=" + str(self.numProcesses) + " testSize=" + str(self.testSize)
        
        return outputStr 

    def copy(self): 
        maxLocalAuc = MaxLocalAUC(k=self.k, w=self.w, lmbda=self.lmbda)
        maxLocalAuc.eps = self.eps 
        maxLocalAuc.stochastic = self.stochastic
        maxLocalAuc.rho = self.rho 
     
        maxLocalAuc.rate = self.rate
        maxLocalAuc.alpha = self.alpha
        maxLocalAuc.t0 = self.t0
        
        maxLocalAuc.recordStep = self.recordStep
        maxLocalAuc.numRowSamples = self.numRowSamples
        maxLocalAuc.numStepIterations = self.numStepIterations
        maxLocalAuc.numAucSamples = self.numAucSamples
        maxLocalAuc.maxIterations = self.maxIterations
        maxLocalAuc.initialAlg = self.initialAlg
        
        maxLocalAuc.ks = self.ks
        maxLocalAuc.lmbdas = self.lmbdas
        maxLocalAuc.folds = self.folds
        maxLocalAuc.testSize = self.testSize
        
        return maxLocalAuc
        