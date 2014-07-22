import logging
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
from sandbox.util.MCEvaluator import MCEvaluator

"""
Some useful functions/classes for the recommended stuff 
"""

    
def computeTestF1(args): 
    """
    A simple function for outputing F1 for a learner in conjunction e.g. with 
    parallel model selection. 
    """
    trainX, testX, learner = args 
        
    learner.learnModel(trainX)
    
    testOrderedItems = MCEvaluatorCython.recommendAtk(learner.U, learner.V, learner.recommendSize, trainX)
    f1 = MCEvaluator.f1AtK(SparseUtils.getOmegaListPtr(testX), testOrderedItems, learner.recommendSize) 
    
    try: 
        learnerStr = learner.modelParamsStr()
    except: 
        learnerStr = str(learner) 
        
    logging.debug("F1@" + str(learner.recommendSize) +  ": " + str('%.4f' % f1) + " " + learnerStr)
        
    return f1
    
def computeTestMRR(args): 
    """
    A simple function for outputing F1 for a learner in conjunction e.g. with 
    parallel model selection. 
    """
    trainX, testX, learner = args 
        
    learner.learnModel(trainX)
    
    testOrderedItems = MCEvaluatorCython.recommendAtk(learner.U, learner.V, learner.recommendSize, trainX)
    mrr = MCEvaluator.mrrAtK(SparseUtils.getOmegaListPtr(testX), testOrderedItems, learner.recommendSize) 
    
    try: 
        learnerStr = learner.modelParamsStr()
    except: 
        learnerStr = str(learner) 
        
    logging.debug("MRR@" + str(learner.recommendSize) +  ": " + str('%.4f' % mrr) + " " + learnerStr)
        
    return mrr