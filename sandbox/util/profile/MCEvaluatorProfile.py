

import numpy
import logging
import sys

from apgl.util.ProfileUtils import ProfileUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.SparseUtils import SparseUtils 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class MCEvaluatorProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
    def profilePrecisionAtK(self):
        m = 1000 
        n = 500000 
        r = 30 
        k = m*100
        
        X, U, s, V = SparseUtils.generateSparseLowRank((m,n), r, k, verbose=True)
        mean = X.data.mean()
        X.data[X.data <= mean] = 0
        X.data[X.data > mean] = 1
        
        import sppy 
        X = sppy.csarray(X)
        
        
        ProfileUtils.profile("MCEvaluator.precisionAtK(X, U, V, 10)", globals(), locals())
        
        
        

profiler = MCEvaluatorProfile()
profiler.profilePrecisionAtK()