import numpy
import logging
import sys
from sandbox.util.ProfileUtils import ProfileUtils
from sandbox.recommendation.BprRecommender import BprRecommender
from sandbox.util.SparseUtils import SparseUtils


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class BprRecommenderProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        m = 1000
        n = 500
        self.k = 8 
        self.X = SparseUtils.generateSparseBinaryMatrix((m, n), self.k, csarray=True)
        
        
    def profileLearnModel(self):
        #Profile full gradient descent 
        u = 0.2
        w = 1-u
        eps = 10**-6
        alpha = 0.5
        learner = BprRecommender(self.k)
        learner.maxIterations = 2
        learner.recordStep = 1
        learner.numAucSamples = 5
        print(learner)
        print(self.X.nnz)
                
        ProfileUtils.profile('learner.learnModel(self.X)', globals(), locals())
        
profiler = BprRecommenderProfile()
profiler.profileLearnModel()  