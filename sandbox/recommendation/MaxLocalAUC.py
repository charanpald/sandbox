
import numpy 
import logging
import multiprocessing 
import sppy 
import time
from math import exp
from sandbox.util.SparseUtils import SparseUtils
from sandbox.recommendation.MaxLocalAUCCython import derivativeUi, derivativeVi, updateVApprox, updateUApprox, objectiveApprox, localAUCApprox
from apgl.util.Sampling import Sampling 
from apgl.util.Util import Util 

class MaxLocalAUC(object): 
    """
    Let's try different ways of maximising the local AUC with a penalty term. 
    """
    def __init__(self, lmbda, k, r, sigma=0.05, eps=0.1, stochastic=False): 
        self.lmbda = lmbda
        self.k = k 
        self.r = r
        self.sigma = sigma
        self.eps = eps 
        self.stochastic = stochastic
        
        #Optimal rate doesn't seem to work 
        self.rate = "constant"
        self.alpha = 0.000002
        self.t0 = 100
        
        self.recordStep = 10
        
        self.numRowSamples = 10
        self.numAucSamples = 100
        self.maxIterations = 1000
        self.iterationsPerUpdate = 10
        self.initialAlg = "rand"
        
        #Model selection parameters 
        self.folds = 3 
        self.ks = numpy.array([10, 20, 50])
        self.lmbdas = numpy.array([10**-7, 10**-6, 10**-5, 10**-4]) 
    
    def getOmegaList(self, X): 
        """
        Return a list such that the ith element contains an array of nonzero 
        entries in the ith row of X. 
        """
        omegaList = []
        
        for i in range(X.shape[0]): 
            omegaList.append(numpy.array(X[i, :].nonzero()[1], numpy.uint))
        return omegaList 
    
    def learnModel(self, X, verbose=False): 
        """
        Max local AUC with Frobenius norm penalty. Solve with gradient descent. 
        The input is a sparse array. 
        """
        
        m = X.shape[0]
        n = X.shape[1]
        omegaList = self.getOmegaList(X)
  
        if self.initialAlg == "rand": 
            U = numpy.random.rand(m, self.k)
            V = numpy.random.rand(n, self.k)
        elif self.initialAlg == "ones": 
            U = numpy.ones((m, self.k))
            V = numpy.ones((n, self.k))
        elif self.initialAlg == "svd":
            logging.debug("Initialising with SVD")
            try: 
                U, s, V = SparseUtils.svdPropack(X, self.k)
            except ImportError: 
                U, s, V = SparseUtils.svdArpack(X, self.k)
            U = numpy.ascontiguousarray(U)
            V = numpy.ascontiguousarray(V)
        else:
            raise ValueError("Unknown initialisation: " + str(self.initialAlg))
        
        lastU = U+numpy.ones((m, self.k))*self.eps
        lastV = V+numpy.ones((n, self.k))*self.eps 
        
        normDeltaU = numpy.linalg.norm(U - lastU)
        normDeltaV = numpy.linalg.norm(V - lastV)
        objs = []
        aucs = []
        
        ind = 0
        eps = self.eps 

        #Convert to a csarray for faster access 
        logging.debug("Converting to csarray")
        X2 = sppy.csarray(X.shape, storagetype="row")
        X2[X.nonzero()] = X.data
        X2.compress()
        X = X2
        
        startTime = time.time()
    
        while (normDeltaU > eps or normDeltaV > eps) and ind < self.maxIterations: 
            lastU = U.copy() 
            lastV = V.copy() 
            
            if self.rate == "constant": 
                sigma = self.sigma 
            elif self.rate == "optimal":
                sigma = 1/(self.alpha*(self.t0 + ind))
                logging.debug("sigma=" + str(sigma))
            else: 
                raise ValueError("Invalid rate: " + self.rate)

            self.updateU(X, U, V, omegaList)            
            self.updateV(X, U, V, omegaList)

            normDeltaU = numpy.linalg.norm(U - lastU)
            normDeltaV = numpy.linalg.norm(V - lastV)               
                
            if ind % self.recordStep == 0: 
                objs.append(objectiveApprox(X, U, V, omegaList, self.numAucSamples, self.lmbda, self.r))
                aucs.append(localAUCApprox(X, U, V, omegaList, self.numAucSamples, self.r))
                printStr = "Iteration: " + str(ind)
                printStr += " local AUC~" + str(aucs[-1]) + " objective~" + str(objs[-1])
                printStr += " ||dU||=" + str(normDeltaU) + " " + "||dV||=" + str(normDeltaV)
                logging.debug(printStr)
            
            ind += 1
            
        totalTime = time.time() - startTime
        logging.debug("||dU||=" + str(normDeltaU) + " " + "||dV||=" + str(normDeltaV))
        logging.debug("Total time taken " + str(totalTime))
                        
        if verbose:     
            return U, V, numpy.array(objs), numpy.array(aucs), ind, totalTime
        else: 
            return U, V
        
    def updateU(self, X, U, V, omegaList): 
        """
        Find the derivative with respect to V or part of it. 
        """
        if not self.stochastic: 
            dU = numpy.zeros(U.shape)
            for i in range(X.shape[0]): 
                dU[i, :] = self.derivativeUi(X, U, V, omegaList, i)
            U -= self.sigma*dU
            return dU
        else: 
            updateUApprox(X, U, V, omegaList, self.numAucSamples, self.sigma, self.iterationsPerUpdate, self.k, self.lmbda, self.r) 
        
    #@profile
    def derivativeUi(self, X, U, V, omegaList, i): 
        """
        delta phi/delta u_i
        """
        return derivativeUi(X, U, V, omegaList, i, self.k, self.lmbda, self.r)
        
    def updateV(self, X, U, V, omegaList): 
        """
        Find the derivative with respect to V or part of it. 
        """
        if not self.stochastic: 
            dV = numpy.zeros(V.shape)
            for i in range(X.shape[1]): 
                dV[i, :] = self.derivativeVi(X, U, V, omegaList, i)
            V -= self.sigma*dV
            return dV 
        else: 
            updateVApprox(X, U, V, omegaList, self.numRowSamples, self.numAucSamples, self.sigma, self.iterationsPerUpdate, self.k, self.lmbda, self.r) 
           
    def derivativeVi(self, X, U, V, omegaList, i): 
        return derivativeVi(X, U, V, omegaList, i, self.k, self.lmbda, self.r)           
           
    def localAUC(self, X, U, V, omegaList): 
        """
        Compute the local AUC for the score functions UV^T relative to X with 
        quantile vector r. 
        """
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        localAuc = numpy.zeros(X.shape[0]) 
        allInds = numpy.arange(X.shape[1])
        
        for i in range(X.shape[0]): 
            omegai = omegaList[i]
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0                
                
                for p in omegai: 
                    for q in omegaBari: 
                        if Z[i, p] > Z[i, q] and Z[i, p] > self.r[i]: 
                            partialAuc += 1 
                            
                localAuc[i] = partialAuc/float(omegai.shape[0] * omegaBari.shape[0])
        
        localAuc = localAuc.mean()        
        
        return localAuc

    def localAUCApprox(self, X, U, V, omegaList): 
        """
        Compute the estimated local AUC for the score functions UV^T relative to X with 
        quantile vector r. 
        """
        #For now let's compute the full matrix 
        Z = U.dot(V.T)
        
        localAuc = numpy.zeros(X.shape[0]) 
        allInds = numpy.arange(X.shape[1])

        for i in range(X.shape[0]): 
            omegai = omegaList[i]
            omegaBari = numpy.setdiff1d(allInds, omegai, assume_unique=True)
            
            if omegai.shape[0] * omegaBari.shape[0] != 0: 
                partialAuc = 0 

                for j in range(self.numAucSamples):
                    ind = numpy.random.randint(omegai.shape[0]*omegaBari.shape[0])
                    p = omegai[int(ind/omegaBari.shape[0])] 
                    q = omegaBari[ind % omegaBari.shape[0]]   
                    
                    if Z[i, p] > Z[i, q] and Z[i, p] > self.r[i]: 
                        partialAuc += 1 
                            
                localAuc[i] = partialAuc/float(self.numAucSamples)
            
        localAuc = localAuc.mean()        
        
        return localAuc
    
    #@profile
    def objective(self, X, U, V, omegaList):         
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
        obj = 0.5*self.lmbda * (numpy.sum(U**2) + numpy.sum(V**2)) - obj
        
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
        obj = 0.5*self.lmbda * (numpy.sum(U**2) + numpy.sum(V**2)) - obj
        
        return obj 
        
    def modelSelect(self, X): 
        """
        Perform model selection on X and return the best parameters. 
        """
        cvInds = Sampling.randCrossValidation(self.folds, X.nnz)
        localAucs = numpy.zeros((self.ks.shape[0], self.lmbdas.shape[0], len(cvInds)))
        
        logging.debug("Performing model selection")
        for icv, (trainInds, testInds) in enumerate(cvInds):
            Util.printIteration(icv, 1, self.folds, "Fold: ")

            trainX = SparseUtils.submatrix(X, trainInds)
            testX = SparseUtils.submatrix(X, testInds)
            
            for i, k in enumerate(self.ks): 
                for j, lmbda in enumerate(self.lmbdas): 
                    self.k = k 
                    self.lmbda = lmbda 
                    
                    U, V = self.learnModel(trainX)
                    
                    omegaList = self.getOmegaList(testX)
                    localAucs[i, j, icv] = self.localAUCApprox(testX, U, V, omegaList)
        
        meanLocalAucs = numpy.mean(localAucs, 2)
        stdLocalAucs = numpy.std(localAucs, 2)
        
        logging.debug(meanLocalAucs)
        
        k = self.ks[numpy.unravel_index(numpy.argmax(meanLocalAucs), meanLocalAucs.shape)[0]]
        lmbda = self.lmbdas[numpy.unravel_index(numpy.argmax(meanLocalAucs), meanLocalAucs.shape)[1]]
        
        logging.debug("Model parameters: k=" + str(k) + " lambda=" + str(lmbda))
        
        self.k = k 
        self.lmbda = lmbda 