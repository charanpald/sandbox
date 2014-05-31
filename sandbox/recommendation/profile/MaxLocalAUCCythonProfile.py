import numpy
import logging
import sys
from sandbox.util.ProfileUtils import ProfileUtils
from sandbox.recommendation.MaxLocalAUCCython import derivativeViApprox, objectiveApprox
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.Sampling import Sampling
from sandbox.util.SparseUtilsCython import SparseUtilsCython

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class MaxLocalAUCCythonProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        self.m = 1000 
        self.n = 500 
        self.k = 10 
        self.X = SparseUtils.generateSparseBinaryMatrix((self.m, self.n), self.k, csarray=True)
        
        
    def profileObjective(self): 

        k = 10
        U = numpy.random.rand(self.m, k)
        V = numpy.random.rand(self.n, k)
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(self.X)  
        colIndsProbabilities = numpy.ones(colInds.shape[0])
        
        for i in range(self.m): 
            colIndsProbabilities[indPtr[i]:indPtr[i+1]] /= colIndsProbabilities[indPtr[i]:indPtr[i+1]].sum()
            colIndsProbabilities[indPtr[i]:indPtr[i+1]] = numpy.cumsum(colIndsProbabilities[indPtr[i]:indPtr[i+1]])
        
        r = numpy.zeros(self.m)
        lmbda = 0.001
        rho = 1.0 
        numAucSamples = 100
        
        def run(): 
            numRuns = 10
            for i in range(numRuns):   
                objectiveApprox(indPtr, colInds,  indPtr, colInds, U,  V, r, numAucSamples, lmbda, rho, False)        
        
        ProfileUtils.profile('run()', globals(), locals())        
        
    def profileDerivativeVjApprox(self):
        k = 10
        U = numpy.random.rand(self.m, k)
        V = numpy.random.rand(self.n, k)
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(self.X)  
        colIndsProbabilities = numpy.ones(colInds.shape[0])
        
        for i in range(self.m): 
            colIndsProbabilities[indPtr[i]:indPtr[i+1]] /= colIndsProbabilities[indPtr[i]:indPtr[i+1]].sum()
            colIndsProbabilities[indPtr[i]:indPtr[i+1]] = numpy.cumsum(colIndsProbabilities[indPtr[i]:indPtr[i+1]])
        
        j = 3
        numRowSamples = 100
        numAucSamples = 10
        
        r = numpy.zeros(self.m)
        lmbda = 0.001
        C = 0 
        normalise = True
        
        def run(): 
            numRuns = 1
            for i in range(numRuns): 
                for j in range(self.n):     
                    derivativeViApprox(indPtr, colInds,  U,  V,  colIndsProbabilities, j, numRowSamples, numAucSamples, xi, lmbda, C, normalise)
                
        ProfileUtils.profile('run()', globals(), locals())

profiler = MaxLocalAUCCythonProfile()
#profiler.profileDerivativeVjApprox()
profiler.profileObjective()