
import numpy 
import logging
import multiprocessing 
import sppy 
import time
import scipy.sparse
from math import exp
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.recommendation.MaxLocalAUCCython import derivativeUi, derivativeVi, updateVApprox, updateUApprox, objectiveApprox, localAUCApprox, updateV, updateU
from sandbox.util.Sampling import Sampling 
from sandbox.util.Util import Util 
from sandbox.data.Standardiser import Standardiser 

def updateUV(args): 
    X, U, V, omegaList, numRowSamples, numColSamples, numAucSamples, sigma, lmbda, r  = args   
    
    lastU = U.copy()
    lastV = V.copy()
    
    rowInds = numpy.array(numpy.random.randint(X.shape[0], size=numRowSamples), numpy.uint)
    updateUApprox(X, U, V, omegaList, rowInds, numAucSamples, sigma, lmbda, r)
    deltaU = U - lastU

    rowInds = numpy.array(numpy.random.randint(X.shape[0], size=numRowSamples), numpy.uint)
    colInds = numpy.array(numpy.random.randint(X.shape[1], size=numColSamples), numpy.uint)
    updateVApprox(X, U, V, omegaList, rowInds, colInds, numAucSamples, sigma, lmbda, r)
    deltaV = V - lastV
    
    return deltaU, deltaV


def computeObjective(args): 
    numpy.random.seed(21)
    X, omegaList, U, V, maxLocalAuc  = args 
    
    U, V = maxLocalAuc.learnModel(X, U=U, V=V)
    r = SparseUtilsCython.computeR(U, V, 1-maxLocalAuc.u, maxLocalAuc.numAucSamples)
    
    objective = objectiveApprox(X, U, V, omegaList, maxLocalAuc.numAucSamples, maxLocalAuc.getLambda(X), r)
    
    logging.debug("Objective: " + str(objective) + " with t0 = " + str(maxLocalAuc.t0) + " and alpha= " + str(maxLocalAuc.alpha))
    return objective
    
def localAucsRhos(args): 
    trainX, testX, testOmegaList, maxLocalAuc  = args 
    
    (m, n) = trainX.shape
    U = numpy.random.rand(m, maxLocalAuc.k)
    V = numpy.random.rand(n, maxLocalAuc.k)                
           
    localAucs = numpy.zeros(maxLocalAuc.rhos.shape[0])

    for j, rho in enumerate(maxLocalAuc.rhos): 
        maxLocalAuc.rho = rho 
        
        U, V = maxLocalAuc.learnModel(trainX, U=U, V=V)
        
        r = SparseUtilsCython.computeR(U, V, 1-maxLocalAuc.u, maxLocalAuc.numAucSamples)
        localAucs[j] = localAUCApprox(testX, U, V, testOmegaList, maxLocalAuc.numAucSamples, r) 
        logging.debug("Local AUC: " + str(localAucs[j]) + " with k = " + str(maxLocalAuc.k) + " and rho= " + str(maxLocalAuc.rho))
        
    return localAucs
      
class MaxLocalAUC(object): 
    def __init__(self, rho, k, u, sigma=0.05, eps=0.01, stochastic=False, numProcesses=None): 
        """
        Create an object for  maximising the local AUC with a penalty term using the matrix
        decomposition UV.T 
        
        :param rho: The regularisation parameter 
        
        :param k: The rank of matrices U and V
        
        :param u: The quantile for the local AUC 
        
        :param sigma: The learning rate 
        
        :param eps: The termination threshold for ||dU|| and ||dV||
        
        :stochastic: Whether to use stochastic gradient descent or gradient descent 
        """
        self.rho = rho
        self.k = k 
        self.u = u
        self.sigma = sigma
        self.eps = eps 
        self.stochastic = stochastic

        if numProcesses == None: 
            self.numProcesses = multiprocessing.cpu_count()

        self.chunkSize = 1        
        
        #Optimal rate doesn't seem to work 
        self.rate = "constant"
        self.alpha = 0.1 #Initial learning rate 
        self.t0 = 0.1 #Convergence speed - larger means we get to 0 faster
        
        self.nu = 20.0 
        self.nuBar = 1.0
        self.project = True
        
        self.recordStep = 20
        self.numRowSamples = 50
        self.numColSamples = 50
        self.numAucSamples = 100
        self.maxIterations = 1000
        self.initialAlg = "rand"
        
        #Model selection parameters 
        self.folds = 3 
        self.ks = numpy.array([10, 20, 50, 100])
        self.rhos = numpy.flipud(numpy.logspace(-3, -1, 11)*2) 
        
        #Learning rate selection 
        self.alphas = numpy.linspace(0.1, 100.0, 5)
        self.t0s = 10.0**numpy.arange(-5, 0)
    
    def getLambda(self, X): 
        return self.rho/X.shape[0]
    
    def learnModel(self, X, verbose=False, U=None, V=None, testX=None): 
        """
        Max local AUC with Frobenius norm penalty. Solve with gradient descent. 
        The input is a sparse array. 
        """
        
        m = X.shape[0]
        n = X.shape[1]
        omegaList = SparseUtils.getOmegaList(X)
        if testX != None: 
            testOmegaList = SparseUtils.getOmegaList(testX)

        if U==None or V==None:
            if self.initialAlg == "rand": 
                U = numpy.random.rand(m, self.k)
                V = numpy.random.rand(n, self.k)
            elif self.initialAlg == "ones": 
                U = numpy.ones((m, self.k))
                V = numpy.ones((n, self.k))
            elif self.initialAlg == "svd":
                logging.debug("Initialising with SVD")
                try: 
                    U, s, V = SparseUtils.svdPropack(X, self.k, kmax=numpy.min([self.k*15, m-1, n-1]))
                except ImportError: 
                    U, s, V = SparseUtils.svdArpack(X, self.k)
                U = numpy.ascontiguousarray(U)
                V = numpy.ascontiguousarray(V)
            else:
                raise ValueError("Unknown initialisation: " + str(self.initialAlg))
        
        U = Standardiser().normaliseArray(U.T).T    
        V = Standardiser().normaliseArray(V.T).T 
        
        lastU = numpy.random.rand(m, self.k)
        lastV = numpy.random.rand(n, self.k)
        
        normDeltaU = numpy.linalg.norm(U - lastU)
        normDeltaV = numpy.linalg.norm(V - lastV)
        objs = []
        trainAucs = []
        testAucs = []
        
        ind = 0
        r = SparseUtilsCython.computeR(U, V, 1-self.u, self.numAucSamples)

        #Convert to a csarray for faster access 
        if scipy.sparse.issparse(X):
            logging.debug("Converting to csarray")
            X2 = sppy.csarray(X, storagetype="row")
            X = X2
        
        startTime = time.time()
    
        while (normDeltaU > self.eps or normDeltaV > self.eps) and ind < self.maxIterations: 
            lastU = U.copy() 
            lastV = V.copy() 
            
            if self.rate == "constant": 
                pass
            elif self.rate == "optimal":
                self.sigma = self.alpha/((1 + self.alpha*self.t0*ind))
                #logging.debug("sigma=" + str(self.sigma))
            else: 
                raise ValueError("Invalid rate: " + self.rate)
            
            U  = numpy.ascontiguousarray(U)
            r = SparseUtilsCython.computeR(U, V, 1-self.u, self.numAucSamples)
            
            #paramList = []
            #for j in range(self.numProcesses): 
            #    paramList.append((X, U, V, omegaList, self.numRowSamples, self.numColSamples, self.numAucSamples, self.sigma, self.getLambda(X), r))
                
            #pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
            #resultsIterator = pool.imap(updateUV, paramList, self.chunkSize)
            #import itertools
            #resultsIterator = itertools.imap(updateUV, paramList)
            
            #for deltaU, deltaV in resultsIterator: 
            #    U -= deltaU
            #    V -= deltaV
                
            #pool.terminate()
            
            self.updateU(X, U, V, omegaList, r)            
            self.updateV(X, lastU, V, omegaList, r)

            normDeltaU = numpy.linalg.norm(U - lastU)
            normDeltaV = numpy.linalg.norm(V - lastV)               
                
            if ind % self.recordStep == 0: 
                objs.append(objectiveApprox(X, U, V, omegaList, self.numAucSamples, self.getLambda(X), r))
                trainAucs.append(localAUCApprox(X, U, V, omegaList, self.numAucSamples, r))
                
                if testX != None: 
                    testAucs.append(localAUCApprox(testX, U, V, testOmegaList, self.numAucSamples, r))
                printStr = "Iteration: " + str(ind)
                printStr += " local AUC~" + str(trainAucs[-1]) + " objective~" + str(objs[-1])
                printStr += " ||dU||=" + str(normDeltaU) + " " + "||dV||=" + str(normDeltaV)
                logging.debug(printStr)
            
            ind += 1
            
        totalTime = time.time() - startTime
        logging.debug("||dU||=" + str(normDeltaU) + " " + "||dV||=" + str(normDeltaV))
        logging.debug("Total time taken " + str(totalTime))
        logging.debug("Number of iterations: " + str(ind))
                        
        if verbose:     
            return U, V, numpy.array(objs), numpy.array(trainAucs), numpy.array(testAucs), ind, totalTime
        else: 
            return U, V
        
    def updateU(self, X, U, V, omegaList, r): 
        """
        Find the derivative with respect to V or part of it. 
        """
        if not self.stochastic:                 
            lmbda = self.getLambda(X)
            updateU(X, U, V, omegaList, self.k, self.sigma, lmbda, r, self.project)
        else: 
            rowInds = numpy.array(numpy.random.randint(X.shape[0], size=self.numRowSamples), numpy.uint)
            updateUApprox(X, U, V, omegaList, rowInds, self.numAucSamples, self.sigma, self.getLambda(X), r, self.nu, self.nuBar, self.project)
        
    #@profile
    def derivativeUi(self, X, U, V, omegaList, i, r): 
        """
        delta phi/delta u_i
        """
        return derivativeUi(X, U, V, omegaList, i, self.k, self.getLambda(X), r)
        
    def updateV(self, X, U, V, omegaList, r): 
        """
        Find the derivative with respect to V or part of it. 
        """
        if not self.stochastic: 
            lmbda = self.getLambda(X)
            updateV(X, U, V, omegaList, self.k, self.sigma, lmbda, r, self.project)
        else: 
            rowInds = numpy.array(numpy.random.randint(X.shape[0], size=self.numRowSamples), numpy.uint)
            colInds = numpy.array(numpy.random.randint(X.shape[1], size=self.numColSamples), numpy.uint)
            updateVApprox(X, U, V, omegaList, rowInds, colInds, self.numAucSamples, self.sigma, self.getLambda(X), r, self.nu, self.nuBar, self.project)

    def derivativeVi(self, X, U, V, omegaList, i, r): 
        return derivativeVi(X, U, V, omegaList, i, self.k, self.getLambda(X), r)           

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
        obj = 0.5*self.getLambda(X) * (numpy.sum(U**2) + numpy.sum(V**2)) - obj
        
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
        obj = 0.5*self.getLambda(X) * (numpy.sum(U**2) + numpy.sum(V**2)) - obj
        
        return obj 
        
    def learningRateSelect(self, X): 
        """
        Let's set the initial learning rate. 
        """        
        m, n = X.shape
        omegaList = SparseUtils.getOmegaList(X)
        objectives = numpy.zeros((self.t0s.shape[0], self.alphas.shape[0]))
        
        paramList = []      
        
        U = numpy.random.rand(m, self.k)
        V = numpy.random.rand(n, self.k)
        
        U = Standardiser().normaliseArray(U.T).T    
        V = Standardiser().normaliseArray(V.T).T 
                    
        for i, t0 in enumerate(self.t0s): 
            for j, alpha in enumerate(self.alphas): 
                maxLocalAuc = self.copy()
                maxLocalAuc.t0 = t0
                maxLocalAuc.alpha = alpha 
                paramList.append((X, omegaList, U, V, maxLocalAuc))
                    
        pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
        resultsIterator = pool.imap(computeObjective, paramList, self.chunkSize)
        #import itertools
        #resultsIterator = itertools.imap(localAucsRhos, paramList)
        
        for i, t0 in enumerate(self.t0s): 
            for j, alpha in enumerate(self.alphas):  
                objectives[i, j] = resultsIterator.next()
        
        pool.terminate()
                
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
        cvInds = Sampling.randCrossValidation(self.folds, X.nnz)
        localAucs = numpy.zeros((self.ks.shape[0], self.rhos.shape[0], len(cvInds)))
        
        logging.debug("Performing model selection")
        paramList = []        
        
        for icv, (trainInds, testInds) in enumerate(cvInds):
            Util.printIteration(icv, 1, self.folds, "Fold: ")

            trainX = SparseUtils.submatrix(X, trainInds)
            testX = SparseUtils.submatrix(X, testInds)
            
            testOmegaList = SparseUtils.getOmegaList(testX)
            
            for i, k in enumerate(self.ks): 
                maxLocalAuc = self.copy()
                maxLocalAuc.k = k
                paramList.append((trainX, testX, testOmegaList, maxLocalAuc))
                    
        pool = multiprocessing.Pool(processes=self.numProcesses, maxtasksperchild=100)
        resultsIterator = pool.imap(localAucsRhos, paramList, self.chunkSize)
        #import itertools
        #resultsIterator = itertools.imap(localAucsRhos, paramList)
        
        for icv, (trainInds, testInds) in enumerate(cvInds):        
            for i, k in enumerate(self.ks): 
                tempAucs = resultsIterator.next()
                localAucs[i, :, icv] = tempAucs
        
        pool.terminate()
        
        meanLocalAucs = numpy.mean(localAucs, 2)
        stdLocalAucs = numpy.std(localAucs, 2)
        
        logging.debug(meanLocalAucs)
        
        k = self.ks[numpy.unravel_index(numpy.argmax(meanLocalAucs), meanLocalAucs.shape)[0]]
        rho = self.rhos[numpy.unravel_index(numpy.argmax(meanLocalAucs), meanLocalAucs.shape)[1]]
        
        logging.debug("Model parameters: k=" + str(k) + " rho=" + str(rho))
        
        self.k = k 
        self.rho = rho 
        
        return meanLocalAucs, stdLocalAucs
    
    def __str__(self): 
        outputStr = "MaxLocalAUC: rho=" + str(self.rho) + " k=" + str(self.k) + " sigma=" + str(self.sigma) + " eps=" + str(self.eps) 
        outputStr += " stochastic=" + str(self.stochastic) + " numRowSamples=" + str(self.numRowSamples) + " numColSamples=" + str(self.numColSamples)
        outputStr += " numAucSamples=" + str(self.numAucSamples) + " maxIterations=" + str(self.maxIterations) + " initialAlg=" + self.initialAlg
        outputStr += " u=" + str(self.u) + " rate=" + str(self.rate) + " alpha=" + str(self.alpha) + " t0=" + str(self.t0) + " folds=" + str(self.folds)
        
        return outputStr 

    def copy(self): 
        maxLocalAuc = MaxLocalAUC(rho=self.rho, k=self.k, u=self.u)
        maxLocalAuc.sigma = self.sigma
        maxLocalAuc.eps = self.eps 
        maxLocalAuc.stochastic = self.stochastic
     
        maxLocalAuc.rate = self.rate
        maxLocalAuc.alpha = self.alpha
        maxLocalAuc.t0 = self.t0
        
        maxLocalAuc.recordStep = self.recordStep
        maxLocalAuc.numRowSamples = self.numRowSamples
        maxLocalAuc.numColSamples = self.numColSamples
        maxLocalAuc.numAucSamples = self.numAucSamples
        maxLocalAuc.maxIterations = self.maxIterations
        maxLocalAuc.initialAlg = self.initialAlg
        
        maxLocalAuc.rhos = self.rhos
        maxLocalAuc.ks = self.ks
        maxLocalAuc.folds = self.folds
        
        return maxLocalAuc
        