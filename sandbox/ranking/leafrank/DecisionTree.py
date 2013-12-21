
import orange
import orngTree
import numpy
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator 
from apgl.util.Sampling import Sampling 
from sandbox.ranking.leafrank.AbstractOrangePredictor import AbstractOrangePredictor

class DecisionTree(AbstractOrangePredictor):
    def __init__(self, paramDict={}, folds=5, sampleSize=None, numProcesses=1):
        super(DecisionTree, self).__init__()
        self.maxDepth = 10
        self.minSplit = 30
        #Post-pruned using m-error estimate pruning method with parameter m 
        self.m = 2 
        self.paramDict = paramDict
        self.folds = folds 
        self.chunkSize = 2
        self.setMetricMethod("auc2")  
        self.sampleSize = sampleSize  
        self.processes = numProcesses

    def setM(self, m):
        self.m = m

    def getM(self):
        return self.m 

    def setMinSplit(self, minSplit):
        Parameter.checkInt(minSplit, 0, float('inf'))
        self.minSplit = minSplit

    def getMinSplit(self):
        return self.minSplit 

    def setMaxDepth(self, maxDepth):
        Parameter.checkInt(maxDepth, 1, float('inf'))
        self.maxDepth = maxDepth

    def getMaxDepth(self):
        return self.maxDepth 

    def learnModel(self, X, y):
        if numpy.unique(y).shape[0] != 2:
            raise ValueError("Can only operate on binary data")

        classes = numpy.unique(y)
        self.worstResponse = classes[classes!=self.bestResponse][0]

        #We need to convert y into indices
        newY = self.labelsToInds(y)

        XY = numpy.c_[X, newY]
        attrList = []
        for i in range(X.shape[1]):
            attrList.append(orange.FloatVariable("X" + str(i)))

        attrList.append(orange.EnumVariable("y"))
        attrList[-1].addValue(str(self.bestResponse))
        attrList[-1].addValue(str(self.worstResponse))

        self.domain = orange.Domain(attrList)
        eTable = orange.ExampleTable(self.domain, XY)

        #Weight examples and equalise
        #Equalizing computes such weights that the weighted number of examples
        #in each class is equivalent.
        preprocessor = orange.Preprocessor_addClassWeight(equalize=1)
        preprocessor.classWeights = [1-self.weight, self.weight]
        eTable, weightID = preprocessor(eTable)
        eTable.domain.addmeta(weightID, orange.FloatVariable("w"))

        self.learner = orngTree.TreeLearner(m_pruning=self.m, measure="gainRatio")
        self.learner.max_depth = self.maxDepth
        self.learner.stop = orange.TreeStopCriteria_common()
        self.learner.stop.min_instances = self.minSplit
        self.classifier = self.learner(eTable, weightID)

    def getLearner(self):
        return self.learner

    def getClassifier(self):
        return self.classifier 

    def predict(self, X):
        XY = numpy.c_[X, numpy.zeros(X.shape[0])]
        eTable = orange.ExampleTable(self.domain, XY)
        predY = numpy.zeros(X.shape[0])

        for i in range(len(eTable)):
            predY[i] = self.classifier(eTable[i])

        predY = self.indsToLabels(predY)
        return predY
    
    @staticmethod
    def generate(maxDepth=10, minSplit=30, m=2):
        def generatorFunc():
            decisionTree = DecisionTree()
            decisionTree.setMaxDepth(maxDepth)
            decisionTree.setMinSplit(minSplit)
            decisionTree.setM(m)
            return decisionTree
        return generatorFunc

    @staticmethod
    def depth(tree):
        """
        Find the depth of a tree
        """
        if tree == None:
            return 0
        if not tree.branches:
            return 1
        else:
            return max([DecisionTree.depth(branch) for branch in tree.branches]) + 1

    def generateLearner(self, X, y):
        """
        Train using the given examples and labels, and use model selection to
        find the best parameters.
        """
        if numpy.unique(y).shape[0] != 2:
            print(y)
            raise ValueError("Can only operate on binary data")

        #Do model selection first 
        if self.sampleSize == None: 
            idx = Sampling.crossValidation(self.folds, X.shape[0])
            learner, meanErrors = self.parallelModelSelect(X, y, idx, self.paramDict)
        else: 
            idx = Sampling.crossValidation(self.folds, self.sampleSize)
            inds = numpy.random.permutation(X.shape[0])[0:self.sampleSize]
            learner, meanErrors = self.parallelModelSelect(X[inds, :], y[inds], idx, self.paramDict)
            learner = self.getBestLearner(meanErrors, self.paramDict, X, y)
        
        return learner

    def copy(self): 
        learner = DecisionTree()
        learner.maxDepth = self.maxDepth
        learner.minSplit = self.minSplit 
        learner.m = self.m
        learner.paramDict = self.paramDict
        learner.folds = self.folds 
        learner.chunkSize = self.chunkSize 
        learner.sampleSize = self.sampleSize

        return learner     

    def __str__(self):
        outputStr = "DecisionTree: maxDepth=" + str(self.maxDepth) + " minSplit=" + str(self.minSplit) + " m=" + str(self.m) + " sampleSize=" + str(self.sampleSize)
        return outputStr 