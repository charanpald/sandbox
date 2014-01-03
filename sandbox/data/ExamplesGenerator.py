'''
A simple class which can be used to generate test sets of examples. 
'''

#import numpy 
import numpy.random 

class ExamplesGenerator():
    def __init__(self):
        pass
    
    def generateBinaryExamples(self, numExamples=100, numFeatures=10, noise=0.4, verbose=False):
        """
        Generate a certain number of examples with a uniform distribution between 0 and 1. Create
        binary -/+ 1 labels. Must have more than 1 example and feature. 
        """
        if numExamples == 0 or numFeatures == 0: 
            raise ValueError("Cannot generate empty dataset")

        X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        
        y = numpy.sign((X.dot(c)) - numpy.mean(X.dot(c)) + numpy.random.randn(numExamples)*noise)
        y = numpy.array(y, numpy.int)
        
        if not verbose: 
            return X, y
        else: 
            return X, y, c

    def generateRandomBinaryExamples(self, numExamples=100, numFeatures=10):
        """
        Generate a certain number of examples with a uniform distribution between 0 and 1. Create 
        binary -/+ 1 labels
        """ 
        
        X = numpy.random.rand(numExamples, numFeatures)
        y = (numpy.random.rand(numExamples)>0.5)*2 - 1
        
        return X, y
        
        