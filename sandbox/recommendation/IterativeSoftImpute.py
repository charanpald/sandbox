import gc 
import numpy
import logging
import os 
import itertools
import scipy.sparse.linalg
import sandbox.util.SparseUtils as ExpSU
import numpy.testing as nptst 
import multiprocessing 
from sppy import csarray 
from sandbox.misc.RandomisedSVD import RandomisedSVD
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.Util import Util
from sandbox.util.Parameter import Parameter
from sandbox.recommendation.AbstractMatrixCompleter import AbstractMatrixCompleter
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.misc.SVDUpdate import SVDUpdate
from sandbox.util.LinOperatorUtils import LinOperatorUtils
from sandbox.util.SparseUtils import SparseUtils
from sppy.linalg.GeneralLinearOperator import GeneralLinearOperator

def learnPredict(args): 
    """
    A function to train on a training set and test on a test set, for a number 
    of values of rho. 
    """
    learner, trainX, testX, rhos = args 
    logging.debug("k=" + str(learner.getK()))
    logging.debug(learner) 
    
    testInds = testX.nonzero()
    trainXIter = []
    testIndList = []    
    
    for rho in rhos: 
        trainXIter.append(trainX)
        testIndList.append(testInds)
    
    trainXIter = iter(trainXIter)

    ZIter = learner.learnModel(trainXIter, iter(rhos))
    predXIter = learner.predict(ZIter, testIndList)
    
    errors = numpy.zeros(rhos.shape[0])
    for j, predX in enumerate(predXIter): 
        errors[j] = MCEvaluator.rootMeanSqError(testX, predX)
        logging.debug("Error = " + str(errors[j]))
        del predX 
        gc.collect()
        
    return errors 

class IterativeSoftImpute(AbstractMatrixCompleter):
    """
    Given a set of matrices X_1, ..., X_T find the completed matrices.
    """
    def __init__(self, rho=0.1, eps=0.001, k=None, svdAlg="propack", updateAlg="initial", logStep=10, kmax=None, postProcess=False, p=50, q=2, weighted=False, verbose=False, qu=1):
        """
        Initialise imputing algorithm with given parameters. The rho is a value
        for use with the soft thresholded SVD. Eps is the convergence threshold and
        k is the rank of the SVD.

        :param rho: The regularisation parameter for soft-impute in [0, 1] (lambda = rho * maxSv)

        :param eps: The convergence threshold

        :param k: The number of SVs to compute

        :param svdAlg: The algorithm to use for computing a low rank + sparse matrix

        :param updateAlg: The algorithm to use for updating an SVD for a new matrix
        
        :param p: The oversampling used for the randomised SVD
        
        :param q: The exponent used for the randomised SVD 
        """
        super(AbstractMatrixCompleter, self).__init__()

        self.rho = rho
        self.eps = eps
        self.k = k
        self.svdAlg = svdAlg
        self.updateAlg = updateAlg
        self.p = p
        self.q = q
        #The q used for the SVD update 
        self.qu = qu
        if k != None:
            self.kmax = k*5
        else:
            self.kmax = None
        self.logStep = logStep
        self.postProcess = postProcess 
        self.postProcessSamples = 10**6
        self.maxIterations = 30
        self.weighted = weighted 
        self.verbose = verbose 
        self.numProcesses = multiprocessing.cpu_count()
        self.metric = "mse"

    def learnModel(self, XIterator, rhos=None):
        """
        Learn the matrix completion using an iterator which outputs
        a sequence of sparse matrices X. The output of this method is also
        an iterator which outputs a sequence of completed matrices in factorised 
        form. 
        
        :param XIterator: An iterator which emits scipy.sparse.csc_matrix objects 
        
        :param rhos: An optional array of rhos for model selection using warm restarts 
        """

        class ZIterator(object):
            def __init__(self, XIterator, iterativeSoftImpute):
                self.tol = 10**-6
                self.j = 0
                self.XIterator = XIterator
                self.iterativeSoftImpute = iterativeSoftImpute
                self.rhos = rhos 

            def __iter__(self):
                return self
            
            def next(self):
                X = self.XIterator.next()
                logging.debug("Learning on matrix with shape: " + str(X.shape) + " and " + str(X.nnz) + " non-zeros")    
                
                if self.iterativeSoftImpute.weighted: 
                    #Compute row and col probabilities 
                    up, vp = SparseUtils.nonzeroRowColsProbs(X)
                    nzuInds = up==0
                    nzvInds = vp==0
                    u = numpy.sqrt(1/(up + numpy.array(nzuInds, numpy.int))) 
                    v = numpy.sqrt(1/(vp + numpy.array(nzvInds, numpy.int)))
                    u[nzuInds] = 0 
                    v[nzvInds] = 0 
                
                if self.rhos != None: 
                    self.iterativeSoftImpute.setRho(self.rhos.next())

                if not scipy.sparse.isspmatrix_csc(X):
                    raise ValueError("X must be a csc_matrix not " + str(type(X)))
                    
                #Figure out what lambda should be 
                #PROPACK has problems with convergence 
                Y = scipy.sparse.csc_matrix(X, dtype=numpy.float)
                U, s, V = ExpSU.SparseUtils.svdArpack(Y, 1, kmax=20)
                del Y
                #U, s, V = SparseUtils.svdPropack(X, 1, kmax=20)
                maxS = s[0]
                logging.debug("Largest singular value : " + str(maxS))

                (n, m) = X.shape

                if self.j == 0:
                    self.oldU = numpy.zeros((n, 1))
                    self.oldS = numpy.zeros(1)
                    self.oldV = numpy.zeros((m, 1))
                else:
                    oldN = self.oldU.shape[0]
                    oldM = self.oldV.shape[0]

                    if self.iterativeSoftImpute.updateAlg == "initial":
                        if n > oldN:
                            self.oldU = Util.extendArray(self.oldU, (n, self.oldU.shape[1]))
                        elif n < oldN:
                            self.oldU = self.oldU[0:n, :]

                        if m > oldM:
                            self.oldV = Util.extendArray(self.oldV, (m, self.oldV.shape[1]))
                        elif m < oldN:
                            self.oldV = self.oldV[0:m, :]
                    elif self.iterativeSoftImpute.updateAlg == "zero":
                        self.oldU = numpy.zeros((n, 1))
                        self.oldS = numpy.zeros(1)
                        self.oldV = numpy.zeros((m, 1))
                    else:
                        raise ValueError("Unknown SVD update algorithm: " + self.updateAlg)

                rowInds, colInds = X.nonzero()

                gamma = self.iterativeSoftImpute.eps + 1
                i = 0

                self.iterativeSoftImpute.measures = numpy.zeros((self.iterativeSoftImpute.maxIterations, 4))

                while gamma > self.iterativeSoftImpute.eps:
                    if i == self.iterativeSoftImpute.maxIterations: 
                        logging.debug("Maximum number of iterations reached")
                        break 
                    
                    ZOmega = SparseUtilsCython.partialReconstructPQ((rowInds, colInds), self.oldU*self.oldS, self.oldV)
                    Y = X - ZOmega
                    #Y = Y.tocsc()
                    #del ZOmega
                    Y = csarray.fromScipySparse(Y, storagetype="row")
                    gc.collect()
                    
                    #os.system('taskset -p 0xffffffff %d' % os.getpid())

                    if self.iterativeSoftImpute.svdAlg=="propack":
                        L = LinOperatorUtils.sparseLowRankOp(Y, self.oldU, self.oldS, self.oldV, parallel=False)                        
                        newU, newS, newV = SparseUtils.svdPropack(L, k=self.iterativeSoftImpute.k, kmax=self.iterativeSoftImpute.kmax)
                    elif self.iterativeSoftImpute.svdAlg=="arpack":
                        L = LinOperatorUtils.sparseLowRankOp(Y, self.oldU, self.oldS, self.oldV, parallel=False)                        
                        newU, newS, newV = SparseUtils.svdArpack(L, k=self.iterativeSoftImpute.k, kmax=self.iterativeSoftImpute.kmax)
                    elif self.iterativeSoftImpute.svdAlg=="svdUpdate":
                        newU, newS, newV = SVDUpdate.addSparseProjected(self.oldU, self.oldS, self.oldV, Y, self.iterativeSoftImpute.k)
                    elif self.iterativeSoftImpute.svdAlg=="rsvd":
                        L = LinOperatorUtils.sparseLowRankOp(Y, self.oldU, self.oldS, self.oldV, parallel=True)
                        newU, newS, newV = RandomisedSVD.svd(L, self.iterativeSoftImpute.k, p=self.iterativeSoftImpute.p, q=self.iterativeSoftImpute.q)
                    elif self.iterativeSoftImpute.svdAlg=="rsvdUpdate": 
                        L = LinOperatorUtils.sparseLowRankOp(Y, self.oldU, self.oldS, self.oldV, parallel=True)
                        if self.j == 0: 
                            newU, newS, newV = RandomisedSVD.svd(L, self.iterativeSoftImpute.k, p=self.iterativeSoftImpute.p, q=self.iterativeSoftImpute.q)
                        else: 
                            newU, newS, newV = RandomisedSVD.svd(L, self.iterativeSoftImpute.k, p=self.iterativeSoftImpute.p, q=self.iterativeSoftImpute.qu, omega=self.oldV)
                    elif self.iterativeSoftImpute.svdAlg=="rsvdUpdate2":
                        
                        if self.j == 0: 
                            L = LinOperatorUtils.sparseLowRankOp(Y, self.oldU, self.oldS, self.oldV, parallel=True)
                            newU, newS, newV = RandomisedSVD.svd(L, self.iterativeSoftImpute.k, p=self.iterativeSoftImpute.p, q=self.iterativeSoftImpute.q)
                        else: 
                            #Need linear operator which is U s V 
                            L = LinOperatorUtils.lowRankOp(self.oldU, self.oldS, self.oldV)
                            Y = GeneralLinearOperator.asLinearOperator(Y, parallel=True)
                            newU, newS, newV = RandomisedSVD.updateSvd(L, self.oldU, self.oldS, self.oldV, Y, self.iterativeSoftImpute.k, p=self.iterativeSoftImpute.p)
                    else:
                        raise ValueError("Unknown SVD algorithm: " + self.iterativeSoftImpute.svdAlg)

                    if self.iterativeSoftImpute.weighted and i==0: 
                        delta = numpy.diag((u*newU.T).dot(newU))
                        pi = numpy.diag((v*newV.T).dot(newV))
                        lmbda = (maxS/numpy.max(delta*pi))*self.iterativeSoftImpute.rho
                        lmbdav = lmbda*delta*pi
                    elif not self.iterativeSoftImpute.weighted: 
                        lmbda = maxS*self.iterativeSoftImpute.rho
                        if i==0: 
                            logging.debug("lambda: " + str(lmbda))
                        lmbdav = lmbda
                        
                    newS = newS - lmbdav                    
                    #Soft threshold
                    newS = numpy.clip(newS, 0, numpy.max(newS))
                    

                    normOldZ = (self.oldS**2).sum()
                    normNewZmOldZ = (self.oldS**2).sum() + (newS**2).sum() - 2*numpy.trace((self.oldV.T.dot(newV*newS)).dot(newU.T.dot(self.oldU*self.oldS)))

                    #We can get newZ == oldZ in which case we break
                    if normNewZmOldZ < self.tol:
                        gamma = 0
                    elif abs(normOldZ) < self.tol:
                        gamma = self.iterativeSoftImpute.eps + 1
                    else:
                        gamma = normNewZmOldZ/normOldZ
                        
                    if self.iterativeSoftImpute.verbose: 
                        theta1 = (self.iterativeSoftImpute.k - numpy.linalg.norm(self.oldU.T.dot(newU), 'fro')**2)/self.iterativeSoftImpute.k
                        theta2 = (self.iterativeSoftImpute.k - numpy.linalg.norm(self.oldV.T.dot(newV), 'fro')**2)/self.iterativeSoftImpute.k
                        thetaS = numpy.linalg.norm(newS - self.oldS)**2/numpy.linalg.norm(newS)**2
                        self.iterativeSoftImpute.measures[i, :] = numpy.array([gamma, theta1, theta2, thetaS])

                    self.oldU = newU.copy()
                    self.oldS = newS.copy()
                    self.oldV = newV.copy()

                    logging.debug("Iteration " + str(i) + " gamma="+str(gamma))
                    i += 1

                if self.iterativeSoftImpute.postProcess: 
                    #Add the mean vectors 
                    previousS = newS
                    newU = numpy.c_[newU, numpy.array(X.mean(1)).ravel()]
                    newV = numpy.c_[newV, numpy.array(X.mean(0)).ravel()]
                    newS = self.iterativeSoftImpute.unshrink(X, newU, newV)  
                    
                    #Note that this increases the rank of U and V by 1 
                    #print("Difference in s after postprocessing: " + str(numpy.linalg.norm(previousS - newS[0:-1]))) 
                    logging.debug("Difference in s after postprocessing: " + str(numpy.linalg.norm(previousS - newS[0:-1]))) 

                logging.debug("Number of iterations for rho="+str(self.iterativeSoftImpute.rho) + ": " + str(i))
                self.j += 1
                return (newU, newS, newV)

        return ZIterator(XIterator, self)

    def predict(self, ZIter, indList):
        """
        Make a set of predictions for a given iterator of completed matrices and
        an index list.
        """
        class ZTestIter(object):
            def __init__(self, iterativeSoftImpute):
                self.i = 0
                self.iterativeSoftImpute = iterativeSoftImpute

            def __iter__(self):
                return self

            def next(self):    
                Xhat = self.iterativeSoftImpute.predictOne(ZIter.next(), indList[self.i])  
                self.i += 1
                return Xhat 

        return ZTestIter(self)

    def predictOne(self, Z, inds): 
        U, s, V = Z
        
        if type(inds) == tuple: 
            logging.debug("Predicting on matrix with shape: " + str((U.shape[0], V.shape[0])) + " and " + str(inds[0].shape[0]) + " non-zeros")  
        Xhat = ExpSU.SparseUtils.reconstructLowRank(U, s, V, inds)
    
        return Xhat

    def modelSelect(self, X, rhos, ks, cvInds):
        """
        Pick a value of rho based on a single matrix X. We do cross validation
        within, and return the best value of lambda (according to the mean
        squared error). The rhos must be in decreasing order and we use 
        warm restarts. 
        """
        if (numpy.flipud(numpy.sort(rhos)) != rhos).all(): 
            raise ValueError("rhos must be in descending order")    

        errors = numpy.zeros((rhos.shape[0], ks.shape[0], len(cvInds)))

        for i, (trainInds, testInds) in enumerate(cvInds):
            Util.printIteration(i, 1, len(cvInds), "Fold: ")
            trainX = SparseUtils.submatrix(X, trainInds)
            testX = SparseUtils.submatrix(X, testInds)

            assert trainX.nnz == trainInds.shape[0]
            assert testX.nnz == testInds.shape[0]
            nptst.assert_array_almost_equal((testX+trainX).data, X.data)

            paramList = []
        
            for m, k in enumerate(ks): 
                learner = self.copy()
                learner.updateAlg="initial" 
                learner.setK(k)
                paramList.append((learner, trainX, testX, rhos)) 
            
            if self.numProcesses != 1: 
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()/2, maxtasksperchild=10)
                results = pool.imap(learnPredict, paramList)
            else: 
                results = itertools.imap(learnPredict, paramList)
            
            for m, rhoErrors in enumerate(results): 
                errors[:, m, i] = rhoErrors
            
            if self.numProcesses != 1: 
                pool.terminate()

        meanErrors = errors.mean(2)
        stdErrors = errors.std(2)
        
        logging.debug(meanErrors)
        
        #Set the parameters 
        self.setRho(rhos[numpy.unravel_index(numpy.argmin(meanErrors), meanErrors.shape)[0]]) 
        self.setK(ks[numpy.unravel_index(numpy.argmin(meanErrors), meanErrors.shape)[1]])
                
        logging.debug("Model parameters: k=" + str(self.k) + " rho=" + str(self.rho))

        return meanErrors, stdErrors

    def unshrink(self, X, U, V): 
        """
        Perform post-processing on a factorisation of a matrix X use factor 
        vectors U and V. 
        """
        logging.debug("Post processing singular values")
               
        #Fix for versions of numpy < 1.7 
        inds = numpy.unique(numpy.random.randint(0, X.data.shape[0], numpy.min([self.postProcessSamples, X.data.shape[0]]))) 
        a = X.data[inds]
            
        B = numpy.zeros((a.shape[0], U.shape[1])) 
            
        rowInds, colInds = X.nonzero() 
        rowInds = numpy.array(rowInds[inds], numpy.int32)
        colInds = numpy.array(colInds[inds], numpy.int32)  
        
        #Populate B 
        for i in range(U.shape[1]): 
            B[:, i] = SparseUtilsCython.partialOuterProduct(rowInds, colInds, U[:, i], V[:, i])
        
        s = numpy.linalg.pinv(B.T.dot(B)).dot(B.T).dot(a)
        
        return s 

    def setK(self, k):
        #Parameter.checkInt(k, 1, float('inf'))

        self.k = k
        self.kmax = k*5

    def getK(self):
        return self.k

    def setRho(self, rho):
        Parameter.checkFloat(rho, 0.0, 1.0)

        self.rho = rho

    def getRho(self):
        return self.rho

    def getMetricMethod(self):
        return MCEvaluator.meanSqError

    def copy(self):
        """
        Return a new copied version of this object.
        """
        iterativeSoftImpute = IterativeSoftImpute(rho=self.rho, eps=self.eps, k=self.k, svdAlg=self.svdAlg, updateAlg=self.updateAlg, logStep=self.logStep, kmax=self.kmax, postProcess=self.postProcess, weighted=self.weighted, p=self.p, q=self.q)

        return iterativeSoftImpute

    def __str__(self): 
        outputStr = self.name() + ":" 
        outputStr += " rho=" + str(self.rho)+" eps="+str(self.eps)+" k="+str(self.k) + " svdAlg="+str(self.svdAlg) + " kmax="+str(self.kmax)
        outputStr += " postProcess=" + str(self.postProcess) + " weighted="+str(self.weighted) + " p="+str(self.p) + " q="+str(self.q)
        outputStr += " maxIterations=" + str(self.maxIterations)
        return outputStr

    def name(self):
        return "IterativeSoftImpute"
