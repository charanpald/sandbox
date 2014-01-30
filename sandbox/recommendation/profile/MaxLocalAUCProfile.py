import numpy
import logging
import sys
from apgl.util.ProfileUtils import ProfileUtils
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class MaxLocalAUCProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        m = 500 
        n = 200 
        self.k = 5 
        numInds = int(m*n*0.1)
        self.X = SparseUtils.generateSparseLowRank((m, n), self.k, numInds)
        
        self.X = self.X/self.X
        self.X = self.X.tocsr()
        
    def profileLearnModel(self):
        lmbda = 0.00001
        r = numpy.ones(self.X.shape[0])*0.0
        eps = 0.5
        sigma = 100
        maxLocalAuc = MaxLocalAUC(lmbda, self.k, r, sigma=sigma, eps=eps)
                
        ProfileUtils.profile('maxLocalAuc.learnModel(self.X)', globals(), locals())

    def profileLearnModel2(self):
        #Profile stochastic case 
        lmbda = 0.00
        r = numpy.ones(self.X.shape[0])*0.0
        eps = 0.001
        sigma = 10
        maxLocalAuc = MaxLocalAUC(lmbda, self.k, r, sigma=sigma, eps=eps, stochastic=True)
                
        ProfileUtils.profile('maxLocalAuc.learnModel(self.X)', globals(), locals())

profiler = MaxLocalAUCProfile()
#profiler.profileLearnModel()  
profiler.profileLearnModel2()