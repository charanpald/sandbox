import numpy
import logging
import sys
import scipy 
import scipy.sparse 
import scipy.sparse.linalg 
import scipy.io
from sandbox.util.ProfileUtils import ProfileUtils
from exp.sandbox.RandomisedSVD import RandomisedSVD 
from sandbox.util.PathDefaults import PathDefaults 
from exp.recommendexp.NetflixDataset import NetflixDataset
from exp.util.LinOperatorUtils import LinOperatorUtils 
from exp.util.GeneralLinearOperator import GeneralLinearOperator 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class RandomisedSvdProfile(object):
    def __init__(self):
        numpy.random.seed(21)

    def profileSvd(self):
        n = 5000 
        p = 0.1 
        L = scipy.sparse.rand(n, n, p)            
        L = L.T.dot(L)
            
        k = 50 
        q = 2
        ProfileUtils.profile('RandomisedSVD.svd(L, k, q)', globals(), locals())
        
        #Compare against the exact svd - it's much faster 
        #ProfileUtils.profile('scipy.sparse.linalg.svds(L, k=2*k)', globals(), locals())

    def profileSvd2(self):
        dataDir = PathDefaults.getDataDir() + "erasm/contacts/" 
        trainFilename = dataDir + "contacts_train"        
        
        trainX = scipy.io.mmread(trainFilename)
        trainX = scipy.sparse.csc_matrix(trainX, dtype=numpy.int8)
        
        k = 500 
        U, s, V = RandomisedSVD.svd(trainX, k)
        
        print(s)
        
        print("All done")
        
    def profileSvd3(self):
        dataset = NetflixDataset()
        iterator = dataset.getTrainIteratorFunc()
        X = iterator.next() 
        
        #L = LinOperatorUtils.parallelSparseOp(X)  
        L = GeneralLinearOperator.asLinearOperator(X)
        
        k = 50 
        U, s, V = RandomisedSVD.svd(L, k)
        
        print(s)
        
        print("All done")

profiler = RandomisedSvdProfile()
#profiler.profileSvd() #51.4 s
profiler.profileSvd3() 
