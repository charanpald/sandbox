import array
import logging
import multiprocessing 
import numpy 
import scipy.sparse
import sharedmem 
import sppy 
import time
from sandbox.misc.RandomisedSVD import RandomisedSVD
from sandbox.recommendation.AbstractRecommender import AbstractRecommender
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute 
from sandbox.recommendation.MaxAUCTanh import MaxAUCTanh
from sandbox.recommendation.MaxAUCHinge import MaxAUCHinge
from sandbox.recommendation.MaxAUCSquare import MaxAUCSquare
from sandbox.recommendation.MaxAUCLogistic import MaxAUCLogistic
from sandbox.recommendation.MaxAUCSigmoid import MaxAUCSigmoid
from sandbox.recommendation.RecommenderUtils import computeTestMRR, computeTestF1
from sandbox.recommendation.WeightedMf import WeightedMf
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython 
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.Sampling import Sampling 
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.util.SparseUtils import SparseUtils


def computeObjective(args): 
    """
    Compute the objective for a particular parameter set. Used to set a learning rate. 
    """
    X, U, V, maxLocalAuc = args 
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.singleLearnModel(X, U=U, V=V, verbose=True)
    obj = trainMeasures[-1, 0]
    logging.debug("Final objective: " + str(obj) + " with t0=" + str(maxLocalAuc.t0) + " and alpha=" + str(maxLocalAuc.alpha))
    return obj
          
def updateUVBlock(sharedArgs, methodArgs): 
    """
    Compute the objective for a particular parameter set. Used to set a learning rate. 
    """
    rowIsFree, colIsFree, iterationsPerBlock, gradientsPerBlock, U, V, muU, muV, lock = sharedArgs
    learner, rowBlockSize, colBlockSize, indPtr, colInds, permutedRowInds, permutedColInds, gi, gp, gq, normGp, normGq, pid, loopInd, omegasList = methodArgs

    while (iterationsPerBlock < learner.parallelStep).any(): 
        #Find free block 
        lock.acquire()
        inds = numpy.argsort(numpy.ravel(iterationsPerBlock))
        foundBlock = False        
        
        #Find the block with smallest number of updates which is free 
        for i in inds: 
            rowInd, colInd = numpy.unravel_index(i, iterationsPerBlock.shape)
            
            if rowIsFree[rowInd] and colIsFree[colInd]: 
                rowIsFree[rowInd] = False
                colIsFree[colInd] = False
                foundBlock = True
                break
        
        blockRowInds = permutedRowInds[rowInd*rowBlockSize:(rowInd+1)*rowBlockSize]
        blockColInds = permutedColInds[colInd*colBlockSize:(colInd+1)*colBlockSize]
        
        ind = iterationsPerBlock[rowInd, colInd] + loopInd
        sigma = learner.getSigma(ind)
          
        lock.release()
    
        #Now update U and V based on the block 
        if foundBlock: 
            ind = iterationsPerBlock[rowInd, colInd] + loopInd
            sigma = learner.getSigma(ind)
            numIterations = gradientsPerBlock[rowInd, colInd]
            
            indPtr2, colInds2 = omegasList[colInd]

            learner.updateUV(indPtr2, colInds2, U, V, muU, muV, blockRowInds, blockColInds, gp, gq, normGp, normGq, ind, sigma, numIterations)
        else: 
            time.sleep(3)

        lock.acquire()
        if foundBlock:
            rowIsFree[rowInd] = True 
            colIsFree[colInd] = True
            iterationsPerBlock[rowInd, colInd] += 1
        lock.release()
        
def restrictOmega(indPtr, colInds, colIndsSubset): 
    """
    Take a set of nonzero indices for a matrix and restrict the columns to colIndsSubset. 
    """
    m = indPtr.shape[0]-1
    newIndPtr = numpy.zeros(indPtr.shape[0], indPtr.dtype)
    newColInds = array.array("I")
    ptr = 0 
    
    for i in range(m): 
        omegai = colInds[indPtr[i]:indPtr[i+1]]
        newOmegai = numpy.intersect1d(omegai, colIndsSubset)
        
        newIndPtr[i] = ptr 
        newIndPtr[i+1] = ptr + newOmegai.shape[0]
        newColInds.extend(newOmegai)
        ptr += newOmegai.shape[0]
       
    newColInds = numpy.array(newColInds, dtype=colInds.dtype)   
    return newIndPtr, newColInds
      
class MaxLocalAUC(AbstractRecommender): 
    def __init__(self, k, w, alpha=0.05, eps=10**-4, lmbdaU=0, lmbdaV=1, maxIterations=50, stochastic=False, numProcesses=None): 
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
        
        
        self.alpha = alpha #Initial learning rate 
        self.beta = 0.75
        self.bound = False
        self.delta = 0.05
        self.eps = eps 
        self.eta = 5
        self.initialAlg = "rand"
        self.itemExpP = 0.0 #Sample from power law between 0 and 1 
        self.itemExpQ = 0.5     
        self.itemFactors = False
        self.k = k 
        self.lmbdaU = lmbdaU 
        self.lmbdaV = lmbdaV 
        self.maxIterations = maxIterations
        self.maxNorm = 100
        self.maxNormTanh = 10  #Max norm for tahn losses 
        self.metric = "f1"
        self.normalise = True
        self.numAucSamples = 10
        self.numRecordAucSamples = 100
        self.numRowSamples = 30
        self.numRuns = 200
        self.loss = "hinge" 
        self.p = 10 
        self.parallelSGD = False
        self.parallelStep = 5
        self.q = 3
        self.rate = "constant"
        self.recordStep = 10
        self.reg = True
        self.rho = 1.0
        self.startAverage = 30
        self.stochastic = stochastic
        self.t0 = 0.1 #Convergence speed - larger means we get to 0 faster
        self.validationUsers = 0.1
        self.w = w
               
        #Model selection parameters 
        self.ks = 2**numpy.arange(3, 8)
        self.lmbdas = 10.0**numpy.arange(-0.5, 1.5, 0.25)
        self.rhos = numpy.array([0, 0.1, 0.5, 1.0])
        self.itemExps = numpy.array([0, 0.25, 0.5, 0.75, 1.0])
        
        #Learning rate selection 
        self.t0s = 2.0**-numpy.arange(0.0, 5.0)
        self.alphas = 2.0**-numpy.arange(-2.0, 4.0)
    
    def __str__(self): 
        outputStr = "MaxLocalAUC: "

        attributes = vars(self)
        
        for key, item in sorted(attributes.iteritems()):
            if isinstance(item, int) or isinstance(item, float) or isinstance(item, str) or isinstance(item, bool): 
                outputStr += key + "=" + str(item) + " " 
        
        return outputStr     
    
    def computeBound(self, X, U, V, trainExp, delta): 
        """
        Compute a lower bound on the expectation of the loss based on Rademacher 
        theory.
        """
        m, n = X.shape
        Ru = numpy.linalg.norm(U)
        Rv = numpy.linalg.norm(V)
        
        X = X.toarray()
        Xs = X.sum(1)
        E = (X.T / Xs).T
        
        EBar = numpy.ones(X.shape) - X
        EBar = (EBar.T / (EBar.sum(1))).T
        
        P, sigmaM, Q = numpy.linalg.svd(E - EBar)
        sigma1 = numpy.max(sigmaM)
        
        omegaSum = 0  
        
        for i in range(m): 
            omegaSum += 1.0/(Xs[i] * (n-Xs[i])**2)
            
        if self.loss in ["hinge", "square"]: 
            B = 4 
        elif self.loss in ["logistic", "sigmoid"]: 
            B = 1
        else: 
            raise ValueError("Unsupported loss: " + self.loss)
        
        rademacherTerm = 2*B*Ru*Rv*sigma1/m  + numpy.sqrt((2*numpy.log(1/delta)*(n-1)**2/m**2) * omegaSum)     
        secondTerm = numpy.sqrt((numpy.log(1/delta)*(n-1)**2/(2*m**2)) * omegaSum)  
        expectationBound = trainExp + rademacherTerm + secondTerm 
        
        #print(B,Ru,Rv,m,sigma1 )
        #print(trainExp, rademacherTerm, secondTerm)
        
        return expectationBound
    
    def computeGipq(self, X): 
        m, n = X.shape 
        gi = numpy.ones(m)/float(m)
        itemProbs = (X.sum(0)+1)/float(m+1)
        gp = itemProbs**self.itemExpP 
        gp /= gp.sum()
        gq = (1-itemProbs)**self.itemExpQ 
        gq /= gq.sum()
        
        return gi, gp, gq    

    def computeNormGpq(self, indPtr, colInds, gp, gq, m):
        
        normGp = numpy.zeros(m)
        normGq = numpy.zeros(m) 
        gqSum = gq.sum()
        
        for i in range(m):
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            normGp[i] = gp[omegai].sum()
            normGq[i] = gqSum - gq[omegai].sum()
            
        return normGp, normGq    

    def copy(self): 
        maxLocalAuc = MaxLocalAUC(k=self.k, w=self.w, lmbdaU=self.lmbdaU, lmbdaV=self.lmbdaV)
        self.copyParams(maxLocalAuc)

        maxLocalAuc.__dict__.update(self.__dict__)
                            
        return maxLocalAuc    
        

    def getCythonLearner(self): 
        
        if self.loss == "tanh": 
            learnerCython = MaxAUCTanh(self.k, self.lmbdaU, self.lmbdaV, self.normalise, self.numAucSamples, self.numRowSamples, self.startAverage, self.rho)
        elif self.loss == "hinge": 
            learnerCython = MaxAUCHinge(self.k, self.lmbdaU, self.lmbdaV, self.normalise, self.numAucSamples, self.numRowSamples, self.startAverage, self.rho)
        elif self.loss == "square": 
            learnerCython = MaxAUCSquare(self.k, self.lmbdaU, self.lmbdaV, self.normalise, self.numAucSamples, self.numRowSamples, self.startAverage, self.rho)
        elif self.loss == "logistic": 
            learnerCython = MaxAUCLogistic(self.k, self.lmbdaU, self.lmbdaV, self.normalise, self.numAucSamples, self.numRowSamples, self.startAverage, self.rho)
        elif self.loss == "sigmoid": 
            learnerCython = MaxAUCSigmoid(self.k, self.lmbdaU, self.lmbdaV, self.normalise, self.numAucSamples, self.numRowSamples, self.startAverage, self.rho)
        else: 
            raise ValueError("Unknown objective: " + self.loss)
    
        learnerCython.eta = self.eta      
        
        if self.loss == "tanh": 
            learnerCython.maxNorm = self.maxNormTanh
        else: 
            learnerCython.maxNorm = self.maxNorm
            
        return learnerCython
        
    def getSigma(self, ind): 
        if self.rate == "constant": 
            sigma = self.alpha 
        elif self.rate == "optimal":
            #t0 = self.lmbdaV  
            t0 = self.t0
            
            sigma = self.alpha/((1 + self.alpha*t0*ind)**self.beta)
        else: 
            raise ValueError("Invalid rate: " + self.rate)
            
        return sigma     
    
    def initUV(self, X): 
        m = X.shape[0]
        n = X.shape[1]        
        
        if self.initialAlg == "rand": 
            U = numpy.random.randn(m, self.k)*0.1
            V = numpy.random.randn(n, self.k)*0.1
        elif self.initialAlg == "svd":
            logging.debug("Initialising with Randomised SVD")
            U, s, V = RandomisedSVD.svd(X, self.k, self.p, self.q)
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

        
    def learnModel(self, X, verbose=False, U=None, V=None, randSeed=None):
        if randSeed != None: 
            logging.warn("Seeding random number generator")   
            numpy.random.seed(randSeed)        
        
        if self.parallelSGD: 
            return self.parallelLearnModel(X, verbose, U, V)
        else: 
            return self.singleLearnModel(X, verbose, U, V)
        
    def learningRateSelect(self, X): 
        """
        Let's set the initial learning rate. 
        """        
        m, n = X.shape

        #Constant rate ignores t0 
        if self.rate == "constant": 
            self.t0s = numpy.array([1.0])    

        numInitialUVs = self.folds
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
        objectives = numpy.zeros((self.t0s.shape[0], self.alphas.shape[0], numInitialUVs))
        
        paramList = []   
        logging.debug("t0s=" + str(self.t0s))
        logging.debug("alphas=" + str(self.alphas))
        logging.debug(self)
        
        for k in range(numInitialUVs):
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
        
        for k in range(numInitialUVs):
            for i, t0 in enumerate(self.t0s): 
                for j, alpha in enumerate(self.alphas):  
                    objectives[i, j, k] += resultsIterator.next()
            
        if self.numProcesses != 1: 
            pool.terminate()
            
        meanObjs = numpy.mean(objectives, 2) 
        stdObjs = numpy.std(objectives, 2) 
        logging.debug("t0s=" + str(self.t0s))
        logging.debug("alphas=" + str(self.alphas))
        logging.debug("meanObjs=" + str(meanObjs))
        logging.debug("stdObjs=" + str(stdObjs))
        
        t0 = self.t0s[numpy.unravel_index(numpy.argmin(meanObjs), meanObjs.shape)[0]]
        alpha = self.alphas[numpy.unravel_index(numpy.argmin(meanObjs), meanObjs.shape)[1]]
        
        logging.debug("Learning rate parameters: t0=" + str(t0) + " alpha=" + str(alpha))
        
        self.t0 = t0 
        self.alpha = alpha 
        
        return meanObjs, stdObjs  


    def modelParamsStr(self): 
        outputStr = " lmbdaU=" + str(self.lmbdaU) + " lmbdaV=" + str(self.lmbdaV) + " k=" + str(self.k) + " rho=" + str(self.rho)  + " alpha=" + str(self.alpha)
        return outputStr 

    def modelSelect(self, X, colProbs=None, testX=None): 
        """
        Perform model selection on X and return the best parameters. 
        """
        m, n = X.shape
        if testX==None:
            trainTestXs = Sampling.shuffleSplitRows(X, self.folds, self.validationSize, colProbs=colProbs)
        else: 
            trainTestXs = [[X, testX]]

        #Constant rate ignores t0 
        if self.rate == "constant": 
            self.t0s = numpy.array([1.0])            
            
        testMetrics = numpy.zeros((self.t0s.shape[0], self.ks.shape[0], self.lmbdas.shape[0], self.alphas.shape[0], len(trainTestXs)))
        
        logging.debug("Performing model selection")
        paramList = []        
        
        for i, k in enumerate(self.ks): 
            self.k = k
            
            for icv, (trainX, testX) in enumerate(trainTestXs):
                U, V = self.initUV(trainX)
                for j, lmbda in enumerate(self.lmbdas): 
                    for s, alpha in enumerate(self.alphas): 
                        for t, t0 in enumerate(self.t0s):
                            maxLocalAuc = self.copy()
                            maxLocalAuc.k = k    
                            maxLocalAuc.lmbdaU = lmbda
                            maxLocalAuc.lmbdaV = lmbda
                            maxLocalAuc.alpha = alpha 
                            maxLocalAuc.t0 = t0 
                        
                            paramList.append((trainX, testX, maxLocalAuc))
            
        logging.debug("Set parameters")
        if self.metric == "mrr":
            evaluationMethod = computeTestMRR
        elif self.metric == "f1": 
            evaluationMethod = computeTestF1
        else: 
            raise ValueError("Invalid metric: " + self.metric)
            
        if self.parallelSGD: 
            numProcesses = 1
        else:
            numProcesses = self.numProcesses
        
        if numProcesses != 1: 
            pool = multiprocessing.Pool(processes=numProcesses, maxtasksperchild=100)
            resultsIterator = pool.imap(evaluationMethod, paramList, self.chunkSize)
        else: 
            import itertools
            resultsIterator = itertools.imap(evaluationMethod, paramList)
        
        for i, k in enumerate(self.ks):
            for icv in range(len(trainTestXs)): 
                for j, lmbda in enumerate(self.lmbdas): 
                    for s, alpha in enumerate(self.alphas): 
                        for t, t0 in enumerate(self.t0s):
                            testMetrics[t, i, j, s, icv] = resultsIterator.next()
        
        if numProcesses != 1: 
            pool.terminate()
        
        meanTestMetrics = numpy.mean(testMetrics, 4)
        stdTestMetrics = numpy.std(testMetrics, 4)
        
        logging.debug("t0s=" + str(self.t0s)) 
        logging.debug("ks=" + str(self.ks)) 
        logging.debug("lmbdas=" + str(self.lmbdas)) 
        logging.debug("alphas=" + str(self.alphas))         
        logging.debug("Mean metrics =" + str(meanTestMetrics))
        logging.debug("Std metrics =" + str(stdTestMetrics))
        
        self.t0 = self.t0s[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[0]]
        self.k = self.ks[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[1]]
        self.lmbdaU = self.lmbdas[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[2]]
        self.lmbdaV = self.lmbdas[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[2]]
        self.alpha = self.alphas[numpy.unravel_index(numpy.argmax(meanTestMetrics), meanTestMetrics.shape)[3]]
        
        logging.debug("Model parameters: k=" + str(self.k) + " lmbdaU=" + str(self.lmbdaU) + " lmbdaV=" + str(self.lmbdaV) + " alpha=" + str(self.alpha) + " t0=" + str(self.t0) +  " max=" + str(numpy.max(meanTestMetrics)))
         
        return meanTestMetrics, stdTestMetrics

    def modelSelectRandom(self, X, colProbs=None, testX=None): 
        """
        Perform model selection on X and return the best parameters. 
        """
        m, n = X.shape
        
        if testX==None:
            trainTestXs = Sampling.shuffleSplitRows(X, self.folds, self.validationSize, colProbs=colProbs)
        else: 
            trainTestXs = [[X, testX]]

        #Constant rate ignores t0 
        if self.rate == "constant": 
            self.t0s = numpy.array([1.0])            
            
        testMetrics = numpy.zeros((self.numRuns, len(trainTestXs)))
        
        logging.debug("Performing model selection")
        paramList = []        
        
        for i in range(self.numRuns): 
            maxLocalAuc = self.copy()
            maxLocalAuc.k = numpy.random.choice(self.ks)  
            
            for icv, (trainX, testX) in enumerate(trainTestXs):
                U, V = self.initUV(trainX)

                maxLocalAuc.lmbdaU = numpy.random.choice(self.lmbdas) 
                maxLocalAuc.lmbdaV = numpy.random.choice(self.lmbdas) 
                maxLocalAuc.alpha = numpy.random.choice(self.alphas)  
                maxLocalAuc.t0 = numpy.random.choice(self.t0s)  
                maxLocalAuc.itemExpP = numpy.random.choice(self.itemExps) 
                maxLocalAuc.itemExpQ = numpy.random.choice(self.itemExps)
            
                paramList.append((trainX, testX, maxLocalAuc))
            
        logging.debug("Set parameters")
        if self.metric == "mrr":
            evaluationMethod = computeTestMRR
        elif self.metric == "f1": 
            evaluationMethod = computeTestF1
        else: 
            raise ValueError("Invalid metric: " + self.metric)
            
        if self.parallelSGD: 
            numProcesses = 1
        else:
            numProcesses = self.numProcesses
        
        if numProcesses != 1: 
            pool = multiprocessing.Pool(processes=numProcesses, maxtasksperchild=100)
            resultsIterator = pool.imap(evaluationMethod, paramList, self.chunkSize)
        else: 
            import itertools
            resultsIterator = itertools.imap(evaluationMethod, paramList)
        
        for i in range(self.numRuns): 
            for icv, (trainX, testX) in enumerate(trainTestXs):
                testMetrics[i, icv] = resultsIterator.next()
        
        if numProcesses != 1: 
            pool.terminate()
        
        meanTestMetrics = numpy.mean(testMetrics, 1)
        stdTestMetrics = numpy.std(testMetrics, 1)
        
        logging.debug("t0s=" + str(self.t0s)) 
        logging.debug("ks=" + str(self.ks)) 
        logging.debug("lmbdas=" + str(self.lmbdas)) 
        logging.debug("alphas=" + str(self.alphas))  
        logging.debug("itemExps=" + str(self.itemExps))
        
        bestLearner =   paramList[numpy.argmax(meanTestMetrics)][2]      
        
        self.t0 = bestLearner.t0
        self.k = bestLearner.k
        self.lmbdaU = bestLearner.lmbdaU
        self.lmbdaV = bestLearner.lmbdaV
        self.alpha = bestLearner.alpha
        self.itemExpP = bestLearner.itemExpP
        self.itemExpQ = bestLearner.itemExpQ
        
        logging.debug("Model parameters: k=" + str(self.k) + " lmbdaU=" + str(self.lmbdaU) + " lmbdaV=" + str(self.lmbdaV) + " alpha=" + str(self.alpha) + " t0=" + str(self.t0) + " itemExpP=" + str(self.itemExpP) + " itemExpQ=" + str(self.itemExpQ) +  " max=" + str(numpy.max(meanTestMetrics)))
         
        return meanTestMetrics, stdTestMetrics

    def objectiveApprox(self, positiveArray, U, V, r, gi, gp, gq, allArray=None, full=False): 
        """
        Compute the estimated local AUC for the score functions UV^T relative to X with 
        quantile w. The AUC is computed using positiveArray which is a tuple (indPtr, colInds)
        assuming allArray is None. If allArray is not None then positive items are chosen 
        from positiveArray and negative ones are chosen to complement allArray.
        """
        
        indPtr, colInds = positiveArray
        U = numpy.ascontiguousarray(U)
        V = numpy.ascontiguousarray(V)        
        
        if allArray == None: 
            return self.learnerCython.objectiveApprox(indPtr, colInds, indPtr, colInds, U,  V, gp, gq, full=full, reg=self.reg)         
        else:
            allIndPtr, allColInds = allArray
            return self.learnerCython.objectiveApprox(indPtr, colInds, allIndPtr, allColInds, U,  V, gp, gq, full=full, reg=self.reg)
     

    def parallelLearnModel(self, X, verbose=False, U=None, V=None): 
        """
        Max local AUC with Frobenius norm penalty on V. Solve with parallel (stochastic) gradient descent. 
        The input is a sparse array. 
        """
        #Convert to a csarray for faster access 
        if scipy.sparse.issparse(X):
            logging.debug("Converting to csarray")
            X2 = sppy.csarray(X, storagetype="row")
            X = X2        
        
        m, n = X.shape  
        
        #We keep a validation set in order to determine when to stop 
        if self.validationUsers != 0: 
            numValidationUsers = int(m*self.validationUsers)
            trainX, testX, rowSamples = Sampling.shuffleSplitRows(X, 1, self.validationSize, numRows=numValidationUsers)[0] 
            testIndPtr, testColInds = SparseUtils.getOmegaListPtr(testX)
        else: 
            trainX = X 
            testX = None 
            rowSamples = None
            testIndPtr, testColInds = None, None         
        
        #Not that to compute the test AUC we pick i \in X and j \notin X \cup testX       
        indPtr, colInds = SparseUtils.getOmegaListPtr(trainX)
        allIndPtr, allColInds = SparseUtils.getOmegaListPtr(X)

        if U==None or V==None:
            U, V = self.initUV(trainX)
            
        if self.metric == "f1": 
            metricInd = 2 
        elif self.metric == "mrr": 
            metricInd = 3 
        else: 
            raise ValueError("Unknown metric: " + self.metric)
        
        bestMetric = 0 
        bestU = 0 
        bestV = 0
        trainMeasures = []
        testMeasures = []        
        loopInd = 0
           
        numBlocks = self.numProcesses+1 
        gi, gp, gq = self.computeGipq(X)
        normGp, normGq = self.computeNormGpq(indPtr, colInds, gp, gq, m)        
        
        #Some shared variables
        rowIsFree = sharedmem.ones(numBlocks, dtype=numpy.bool)
        colIsFree = sharedmem.ones(numBlocks, dtype=numpy.bool)
        
        #Create shared factors 
        U2 = sharedmem.zeros((m, self.k))
        V2 = sharedmem.zeros((n, self.k))
        muU2 = sharedmem.zeros((m, self.k))
        muV2 = sharedmem.zeros((n, self.k))
        
        U2[:] = U[:]
        V2[:] = V[:]
        muU2[:] = U[:]
        muV2[:] = V[:]
        del U, V
        
        rowBlockSize = int(numpy.ceil(float(m)/numBlocks))
        colBlockSize = int(numpy.ceil(float(n)/numBlocks))
        
        lock = multiprocessing.Lock()        
        startTime = time.time()
        loopInd = 0
        iterationsPerBlock = sharedmem.zeros((numBlocks, numBlocks))
        
        self.learnerCython = self.getCythonLearner()
        nextRecord = 0 

        while loopInd < self.maxIterations: 
            if loopInd >= nextRecord: 
                if loopInd != 0: 
                    print("")  
                    
                printStr = self.recordResults(muU2, muV2, trainMeasures, testMeasures, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, gi, gp, gq, trainX, startTime)    
                logging.debug(printStr) 
                                
                if testIndPtr != None and testMeasures[-1][metricInd] >= bestMetric: 
                    bestMetric = testMeasures[-1][metricInd]
                    bestU = muU2.copy() 
                    bestV = muV2.copy() 
                elif testIndPtr == None: 
                    bestU = muU2.copy() 
                    bestV = muV2.copy()  
                    
                nextRecord += self.recordStep
            
            iterationsPerBlock = sharedmem.zeros((numBlocks, numBlocks))
            self.parallelUpdateUV(X, U2, V2, muU2, muV2, numBlocks, rowBlockSize, colBlockSize, rowIsFree, colIsFree, indPtr, colInds, lock, gi, gp, gq, normGp, normGq, iterationsPerBlock, loopInd)    
            loopInd += numpy.floor(iterationsPerBlock.mean())

        totalTime = time.time() - startTime
        
        #Compute quantities for last U and V 
        print("")
        totalTime = time.time() - startTime
        printStr = "Finished, time=" + str('%.1f' % totalTime) + " "
        printStr += self.recordResults(muU2, muV2, trainMeasures, testMeasures, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, gi, gp, gq, trainX, startTime)
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

    def parallelUpdateUV(self, X, U, V, muU, muV, numBlocks, rowBlockSize, colBlockSize, rowIsFree, colIsFree, indPtr, colInds, lock, gi, gp, gq, normGp, normGq, iterationsPerBlock, loopInd):
        m, n = X.shape
        gradientsPerBlock = sharedmem.zeros((numBlocks, numBlocks))
        
        #Set up order of indices for stochastic methods 
        permutedRowInds = numpy.array(numpy.random.permutation(m), numpy.uint32)
        permutedColInds = numpy.array(numpy.random.permutation(n), numpy.uint32)

        for i in range(numBlocks): 
            for j in range(numBlocks): 
                blockRowInds = numpy.sort(numpy.array(permutedRowInds[i*rowBlockSize:(i+1)*rowBlockSize], numpy.int))
                blockColInds = numpy.sort(numpy.array(permutedColInds[j*colBlockSize:(j+1)*colBlockSize], numpy.int))  
                block = X[blockRowInds, :][:, blockColInds]
                
                gradientsPerBlock[i,j] = max(numpy.ceil(float(block.nnz)/self.numAucSamples), 1)
        
        assert gradientsPerBlock.sum() >= X.nnz/self.numAucSamples

        #Compute omega for each col block 
        omegasList = []
        for i in range(numBlocks): 
            blockColInds = permutedColInds[i*colBlockSize:(i+1)*colBlockSize]
            omegasList.append(restrictOmega(indPtr, colInds, blockColInds))

        processList = []        
        
        if self.numProcesses != 1: 
            for i in range(self.numProcesses):
                learner = self.copy()
                learner.learnerCython = self.getCythonLearner()
                sharedArgs = rowIsFree, colIsFree, iterationsPerBlock, gradientsPerBlock, U, V, muU, muV, lock 
                methodArgs = learner, rowBlockSize, colBlockSize, indPtr, colInds, permutedRowInds, permutedColInds, gi, gp, gq, normGp, normGq, i, loopInd, omegasList
    
                process = multiprocessing.Process(target=updateUVBlock, args=(sharedArgs, methodArgs))
                process.start()
                processList.append(process)
            
            for process in processList: 
                process.join()
                
        else: 
            learner = self.copy()
            learner.learnerCython = self.getCythonLearner()
            sharedArgs = rowIsFree, colIsFree, iterationsPerBlock, gradientsPerBlock, U, V, muU, muV, lock 
            methodArgs = learner, rowBlockSize, colBlockSize, indPtr, colInds, permutedRowInds, permutedColInds, gi, gp, gq, normGp, normGq, 0, loopInd, omegasList
            updateUVBlock(sharedArgs, methodArgs)
            
    def predict(self, maxItems): 
        return MCEvaluator.recommendAtk(self.U, self.V, maxItems)

    def recordResults(self, muU, muV, trainMeasures, testMeasures, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, gi, gp, gq, trainX, startTime): 
        
        sigma = self.getSigma(loopInd)        
        r = SparseUtilsCython.computeR(muU, muV, self.w, self.numRecordAucSamples)
        objArr = self.objectiveApprox((indPtr, colInds), muU, muV, r, gi, gp, gq, full=True)
        if trainMeasures == None: 
            trainMeasures = []
        trainMeasures.append([objArr.sum(), MCEvaluator.localAUCApprox((indPtr, colInds), muU, muV, self.w, self.numRecordAucSamples, r), time.time()-startTime]) 
        
        printStr = "iter " + str(loopInd) + ":"
        printStr += " sigma=" + str('%.4f' % sigma)
        printStr += " train: obj~" + str('%.4f' % trainMeasures[-1][0]) 
        printStr += " LAUC~" + str('%.4f' % trainMeasures[-1][1])         
        
        if testIndPtr is not None: 
            testMeasuresRow = []
            testMeasuresRow.append(self.objectiveApprox((testIndPtr, testColInds), muU, muV, r, gi, gp, gq, allArray=(allIndPtr, allColInds)))
            testMeasuresRow.append(MCEvaluator.localAUCApprox((testIndPtr, testColInds), muU, muV, self.w, self.numRecordAucSamples, r, allArray=(allIndPtr, allColInds)))
            testOrderedItems = MCEvaluatorCython.recommendAtk(muU, muV, numpy.max(self.recommendSize), trainX)

            printStr += " validation: obj~" + str('%.4f' % testMeasuresRow[0])
            printStr += " LAUC~" + str('%.4f' % testMeasuresRow[1])

            try: 
                for p in self.recommendSize: 
                    f1Array, orderedItems = MCEvaluator.f1AtK((testIndPtr, testColInds), testOrderedItems, p, verbose=True)
                    testMeasuresRow.append(f1Array[rowSamples].mean())
            except: 
                f1Array, orderedItems = MCEvaluator.f1AtK((testIndPtr, testColInds), testOrderedItems, self.recommendSize, verbose=True)
                testMeasuresRow.append(f1Array[rowSamples].mean())

            printStr += " f1@" + str(self.recommendSize) + "=" + str('%.4f' % testMeasuresRow[-1])                    
                   
            try:
                for p in self.recommendSize: 
                    mrr, orderedItems = MCEvaluator.mrrAtK((testIndPtr, testColInds), testOrderedItems, p, verbose=True)
                    testMeasuresRow.append(mrr[rowSamples].mean())
            except: 
                mrr, orderedItems = MCEvaluator.mrrAtK((testIndPtr, testColInds), testOrderedItems, self.recommendSize, verbose=True)
                testMeasuresRow.append(mrr[rowSamples].mean())
                
            printStr += " mrr@" + str(self.recommendSize) + "=" + str('%.4f' % testMeasuresRow[-1])
            testMeasures.append(testMeasuresRow)
                       
            
        printStr += " ||U||=" + str('%.3f' % numpy.linalg.norm(muU))
        printStr += " ||V||=" + str('%.3f' %  numpy.linalg.norm(muV))
        
        if self.bound: 
            trainObj = objArr.sum()

            expectationBound = self.computeBound(trainX, muU, muV, trainObj, self.delta)
            printStr += " bound=" + str('%.3f' %  expectationBound)
            trainMeasures[-1].append(expectationBound)
        
        return printStr
    
    def regularisationPath(self, X): 
        """
        Compute a complete regularisation path for lambda. 
        """
        pass 
        
    
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
        
        m, n = X.shape        
        
        #We keep a validation set in order to determine when to stop 
        if self.validationUsers != 0: 
            numValidationUsers = int(m*self.validationUsers)
            trainX, testX, rowSamples = Sampling.shuffleSplitRows(X, 1, self.validationSize, numRows=numValidationUsers)[0] 
            
            testIndPtr, testColInds = SparseUtils.getOmegaListPtr(testX)
            
            logging.debug("Train X shape and nnz: " + str(trainX.shape) + " " + str(trainX.nnz))    
            logging.debug("Validation X shape and nnz: " + str(testX.shape) + " " + str(testX.nnz))
        else: 
            trainX = X 
            testX = None 
            rowSamples = None
            testIndPtr, testColInds = None, None 

        
        #Note that to compute the test AUC we pick i \in X and j \notin X \cup testX       
        indPtr, colInds = SparseUtils.getOmegaListPtr(trainX)
        allIndPtr, allColInds = SparseUtils.getOmegaListPtr(X)

        if type(U) != numpy.ndarray and type(V) != numpy.ndarray:
            U, V = self.initUV(trainX)
            
        if self.metric == "f1": 
            metricInd = 2 
        elif self.metric == "mrr": 
            metricInd = 3 
        else: 
            raise ValueError("Unknown metric: " + self.metric)
        
        muU = U.copy() 
        muV = V.copy()
        bestMetric = 0 
        bestU = 0 
        bestV = 0
        trainMeasures = []
        testMeasures = []        
        loopInd = 0
        lastObj = 0 
        currentObj = lastObj - 2*self.eps
        
        #Try alternative number of iterations 
        #numIterations = trainX.nnz/self.numAucSamples
        numIterations = max(m, n)
        
        self.learnerCython = self.getCythonLearner()
        
        #Set up order of indices for stochastic methods 
        permutedRowInds = numpy.array(numpy.random.permutation(m), numpy.uint32)
        permutedColInds = numpy.array(numpy.random.permutation(n), numpy.uint32)
        
        startTime = time.time()

        gi, gp, gq = self.computeGipq(X)
        normGp, normGq = self.computeNormGpq(indPtr, colInds, gp, gq, m)
    
        while loopInd < self.maxIterations and abs(lastObj - currentObj) > self.eps: 
            sigma = self.getSigma(loopInd)

            if loopInd % self.recordStep == 0: 
                if loopInd != 0 and self.stochastic: 
                    print("")  
                    
                printStr = self.recordResults(muU, muV, trainMeasures, testMeasures, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, gi, gp, gq, trainX, startTime)    
                logging.debug(printStr) 
                                
                if testIndPtr is not None and testMeasures[-1][metricInd] >= bestMetric: 
                    bestMetric = testMeasures[-1][metricInd]
                    logging.debug("Current best metric=" + str(bestMetric))
                    bestU = muU.copy() 
                    bestV = muV.copy() 
                elif testIndPtr is None: 
                    bestU = muU.copy() 
                    bestV = muV.copy() 

                #Compute objective averaged over last 5 recorded steps 
                trainMeasuresArr = numpy.array(trainMeasures)
                lastObj = currentObj
                currentObj = numpy.mean(trainMeasuresArr[-5:, 0])   
                
            U  = numpy.ascontiguousarray(U)
            self.updateUV(indPtr, colInds, U, V, muU, muV, permutedRowInds, permutedColInds, gp, gq, normGp, normGq, loopInd, sigma, numIterations)                       
            loopInd += 1
            
        #Compute quantities for last U and V 
        totalTime = time.time() - startTime
        printStr = "\nFinished, time=" + str('%.1f' % totalTime) + " "
        printStr += self.recordResults(muU, muV, trainMeasures, testMeasures, loopInd, rowSamples, indPtr, colInds, testIndPtr, testColInds, allIndPtr, allColInds, gi, gp, gq, trainX, startTime)
        
        logging.debug(printStr)        
        logging.debug("Final difference in objectives: " + str(abs(lastObj - currentObj)))
         
        self.U = bestU 
        self.V = bestV
        self.gi = gi
        self.gp = gp
        self.gq = gq
        
        trainMeasures = numpy.array(trainMeasures)
        testMeasures = numpy.array(testMeasures)
         
        if verbose:     
            return self.U, self.V, trainMeasures, testMeasures, loopInd, totalTime
        else: 
            return self.U, self.V    
    
    
    def updateUV(self, indPtr, colInds, U, V, muU, muV, permutedRowInds, permutedColInds, gp, gq, normGp, normGq, ind, sigma, numIterations): 
        """
        Find the derivative with respect to V or part of it. 
        """
        if not self.stochastic:    
            self.learnerCython.updateU(indPtr, colInds, U, V, gp, gq, sigma)
            self.learnerCython.updateV(indPtr, colInds, U, V, gp, gq, sigma)
            
            muU[:] = U[:] 
            muV[:] = V[:]
        else: 
            self.learnerCython.updateUVApprox(indPtr, colInds, U, V, muU, muV, permutedRowInds, permutedColInds, gp, gq, normGp, normGq, ind, numIterations, sigma)
