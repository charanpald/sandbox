
import numpy
import logging
import sys
from sandbox.util.Sampling import Sampling 
from sandbox.util.ProfileUtils import ProfileUtils
from sandbox.util.CythonUtils import inverseChoicePy

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class CythonUtilsProfile(object):
    def __init__(self):
        numpy.random.seed(21)        


    def profileInverseChoice(self):
        
        n = 100000 
        v = numpy.array(numpy.random.choice(n, 100), numpy.int32)
        v = numpy.sort(v)
        
        
        def run(): 
            numRuns = 2000000
            for i in range(numRuns): 
                inverseChoicePy(v, n)
                
        
        ProfileUtils.profile('run()', globals(), locals())
        
profiler = CythonUtilsProfile() 
profiler.profileInverseChoice()