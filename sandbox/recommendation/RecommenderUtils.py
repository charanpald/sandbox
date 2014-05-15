import logging
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
from sandbox.util.MCEvaluator import MCEvaluator

"""
Some useful functions/classes for the recommended stuff 
"""

def computeTestPrecision(args): 
    """
    A simple function for outputing precision for a learner in conjunction e.g. with 
    parallel model selection. 
    """
    trainX, testX, learner = args 
    p = learner.validationSize 
    testOmegaList = SparseUtils.getOmegaList(testX)
        
    learner.learnModel(trainX)
    
    testOrderedItems = MCEvaluatorCython.recommendAtk(learner.U, learner.V, p, trainX)
    precision = MCEvaluator.precisionAtK(testX, testOrderedItems, p, omegaList=testOmegaList)
    logging.debug("Precision@" + str(learner.validationSize) +  ": " + str(precision) + str(learner))
        
    return precision
    
