
import numpy
import logging
import sys
from sandbox.util.Sampling import Sampling
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.ProfileUtils import ProfileUtils

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(22)

class SamplingProfile(object):
    def profileShuffleSplitRows(self):
        m = 10000
        n = 5000
        k = 5 
        u = 0.1
        w = 1-u
        X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)

        k2 = 10
        testSize = 2

        ProfileUtils.profile('Sampling.shuffleSplitRows(X, k2, testSize)', globals(), locals())
        
    def profileSampleUsers(self): 
        m = 10000
        n = 50000
        k = 5 
        u = 0.01
        w = 1-u
        X, U, s, V, wv  = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)

        print(X.nnz)

        k2 = 100000

        ProfileUtils.profile('Sampling.sampleUsers2(X, k2)', globals(), locals())      
        
if __name__ == '__main__':     
    profiler = SamplingProfile()
    #profiler.profileShuffleSplitRows() 
    profiler.profileSampleUsers() 