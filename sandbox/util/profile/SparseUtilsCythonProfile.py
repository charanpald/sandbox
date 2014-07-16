

import numpy
import logging
import sys
import scipy.sparse.linalg
import scipy.io
from sandbox.util.ProfileUtils import ProfileUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults 
from wallhack.rankingexp.DatasetUtils import DatasetUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class SparseUtilsCythonProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
    def profilePartialReconstructValsPQ(self):
        shape = 5000, 10000
        r = 100 
        U, s, V = SparseUtils.generateLowRank(shape, r)
        
        k = 1000000 
        inds = numpy.unravel_index(numpy.random.randint(0, shape[0]*shape[1], k), dims=shape)
        
        ProfileUtils.profile('SparseUtilsCython.partialReconstructValsPQ(inds[0], inds[1], U, V)', globals(), locals())

    def profilePartialReconstructValsPQ2(self):
        shape = 5000, 10000
        r = 100 
        U, s, V = SparseUtils.generateLowRank(shape, r)
        
        k = 1000000 
        inds = numpy.unravel_index(numpy.random.randint(0, shape[0]*shape[1], k), dims=shape)
        
        rowInds = numpy.array(inds[0], numpy.int32)
        colInds = numpy.array(inds[1], numpy.int32)
        
        ProfileUtils.profile('SparseUtilsCython.partialReconstructValsPQ2(rowInds, colInds, U, V)', globals(), locals())


    def computeRProfile(self): 
        X, U, V = DatasetUtils.syntheticDataset1(m=1000, n=20000)
        
        w = 0.9 
        indsPerRow = 50        
        
        numRuns = 1000 
        def run(): 
            for i in range(numRuns): 
                SparseUtilsCython.computeR(U, V, w, indsPerRow)
                
        
        ProfileUtils.profile('run()', globals(), locals())
    
    def profileGenerateSparseBinaryMatrixPL(self): 
        m = 500 
        n = 200 
        k = 10
        density = 0.2
        numpy.random.seed(21)
        #X = SparseUtils.generateSparseBinaryMatrixPL((m,n), k, density=density, csarray=True)   
        
        ProfileUtils.profile('SparseUtilsCython.generateSparseBinaryMatrixPL((m,n), k, density=density, csarray=True)', globals(), locals()) 
    
profiler = SparseUtilsCythonProfile()
#profiler.profilePartialReconstructValsPQ()
#profiler.profilePartialReconstructValsPQ2() #About 10x faster 
profiler.computeRProfile()
