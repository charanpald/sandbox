import multiprocessing

class AbstractRecommender(object): 
    """
    Just some common stuff for all recommenders. 
    """    
    def __init__(self, numProcesses=None): 
        
        #These are parameters used for model selection
        if numProcesses == None: 
            self.numProcesses = multiprocessing.cpu_count()
        
        self.recommendSize = 10
        self.validationSize = 3
        self.folds = 2 
        self.chunkSize = 1 
        
    def copyParams(self, learner): 
        learner.recommendSize = self.recommendSize
        learner.validationSize = self.validationSize
        learner.folds = self.folds 
        learner.numProcesses = self.numProcesses
        learner.chunkSize = self.chunkSize
        
        return learner 
        
    def __str__(self): 
        outputStr = " recommendSize=" + str(self.recommendSize) + " validationSize=" + str(self.validationSize) 
        outputStr += " folds=" + str(self.folds) + " numProcesses=" + str(self.numProcesses)
        outputStr += " chunkSize=" + str(self.chunkSize)

        
        return outputStr 
        
        