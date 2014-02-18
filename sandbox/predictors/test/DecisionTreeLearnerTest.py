import numpy 
import unittest
import numpy.testing as nptst
import logging
from sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
from sandbox.data.ExamplesGenerator import ExamplesGenerator
from sandbox.data.Standardiser import Standardiser    
from sandbox.util.Sampling import Sampling
from sklearn.tree import DecisionTreeRegressor 
from sandbox.predictors.LibSVM import LibSVM
import sklearn.datasets as data 
from sandbox.util.Evaluator import Evaluator
from sklearn import linear_model


class DecisionTreeLearnerTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr("raise")
        self.numExamples = 20
        self.numFeatures = 5
        
        generator = ExamplesGenerator() 
        self.X, self.y = generator.generateBinaryExamples(self.numExamples, self.numFeatures)
        self.y = numpy.array(self.y, numpy.float)
        
        
    def testInit(self): 
        learner = DecisionTreeLearner() 
         
    def testLearnModel(self): 
        #First check the integrety of the trees 
        generator = ExamplesGenerator()         
        
        for i in range(5):        
            numExamples = numpy.random.randint(1, 200)
            numFeatures = numpy.random.randint(1, 10)
            minSplit = numpy.random.randint(1, 50)
            maxDepth = numpy.random.randint(1, 10)
            
            X, y = generator.generateBinaryExamples(numExamples, numFeatures)
            y = numpy.array(y, numpy.float)
        
            learner = DecisionTreeLearner(minSplit=minSplit, maxDepth=maxDepth) 
            learner.learnModel(X, y)        
            tree = learner.getTree() 
            
            for vertexId in tree.getAllVertexIds(): 
                vertex = tree.getVertex(vertexId)
                if vertex.getFeatureInd() != None: 
                    meanValue = y[vertex.getTrainInds()].mean()
                    self.assertEquals(meanValue, vertex.getValue())
                    if tree.isNonLeaf(vertexId): 
                        self.assertTrue(0 <= vertex.getFeatureInd() < X.shape[1]) 
                        self.assertTrue(X[:, vertex.getFeatureInd()].min() <= vertex.getThreshold() <= X[:, vertex.getFeatureInd()].max())
                    self.assertTrue(vertex.getTrainInds().shape[0] >= 1)
            
            
            self.assertTrue(tree.depth() <= maxDepth)
            #Check that each split contains indices from parent 
            root = tree.getRootId()
            vertexStack = [root]
            
            while len(vertexStack) != 0: 
                vertexId = vertexStack.pop()
                neighbours = tree.children(vertexId)
                
                if len(neighbours) > 2: 
                    self.fail("Cannot have more than 2 children") 
                elif len(neighbours) > 0: 
                    inds1 = tree.getVertex(neighbours[0]).getTrainInds()
                    inds2 = tree.getVertex(neighbours[1]).getTrainInds()
                    
                    nptst.assert_array_equal(numpy.union1d(inds1, inds2), numpy.unique(tree.getVertex(vertexId).getTrainInds()))
                    
                    vertexStack.append(neighbours[0])
                    vertexStack.append(neighbours[1])
        
        #Try a tree of depth 0 
        #learner = DecisionTreeLearner(minSplit=10, maxDepth=0) 
        #learner.learnModel(self.X, self.y)        
        #tree = learner.getTree()
        
        #self.assertEquals(tree.depth(), 0)
        
        #Try minSplit > numExamples 
        #learner = DecisionTreeLearner(minSplit=self.numExamples+1, maxDepth=0) 
        #learner.learnModel(self.X, self.y)        
        #tree = learner.getTree()
        
        #self.assertEquals(tree.getNumVertices(), 1)
        
        #Try a simple tree of depth 1 
        learner = DecisionTreeLearner(minSplit=1, maxDepth=1) 
        learner.learnModel(self.X, self.y)     
        
        bestFeature = 0 
        bestError = 10**6 
        bestThreshold = 0         
        
        for i in range(numFeatures): 
            vals = numpy.unique(self.X[:, i])
            
            for j in range(vals.shape[0]-1):             
                threshold = (vals[j+1]+vals[j])/2
                leftInds = self.X[:, i] <= threshold
                rightInds = self.X[:, i] > threshold
                
                valLeft = numpy.mean(self.y[leftInds])
                valRight = numpy.mean(self.y[rightInds])
                
                error = ((self.y[leftInds] - valLeft)**2).sum() + ((self.y[rightInds] - valRight)**2).sum()
                
                if error < bestError: 
                    bestError = error 
                    bestFeature = i 
                    bestThreshold = threshold 
        
        self.assertAlmostEquals(bestThreshold, learner.tree.getRoot().getThreshold())
        self.assertAlmostEquals(bestError, learner.tree.getRoot().getError(), 5)
        self.assertEquals(bestFeature, learner.tree.getRoot().getFeatureInd())
        
        #Now we will test pruning works 
        learner = DecisionTreeLearner(minSplit=1, maxDepth=10) 
        learner.learnModel(X, y)
        numVertices1 = learner.getTree().getNumVertices()       
        
        learner = DecisionTreeLearner(minSplit=1, maxDepth=10, pruneType="REP-CV") 
        learner.learnModel(X, y) 
        numVertices2 = learner.getTree().getNumVertices()   
        
        self.assertTrue(numVertices1 >= numVertices2)
        
    @staticmethod
    def printTree(tree):
        """
        Some code to print the sklearn tree. 
        """
        
        children = tree.children
        
        depth = 0
        nodeIdStack = [(0, depth)] 
         
        
        while len(nodeIdStack) != 0:
            vertexId, depth = nodeIdStack.pop()
            
            if vertexId != tree.LEAF: 
                outputStr = "\t"*depth +str(vertexId) + ": Size: " + str(tree.n_samples[vertexId]) + ", "
                outputStr += "featureInd: " + str(tree.feature[vertexId]) + ", "
                outputStr += "threshold: " + str(tree.threshold[vertexId]) + ", "
                outputStr += "error: " + str(tree.best_error[vertexId]) + ", "
                outputStr += "value: " + str(tree.value[vertexId])
                print(outputStr)
            
                rightChildId = children[vertexId, 1]
                nodeIdStack.append((rightChildId, depth+1))
                
                leftChildId = children[vertexId, 0]
                nodeIdStack.append((leftChildId, depth+1))
        
    def testPredict(self): 
        
        generator = ExamplesGenerator()         
        
        for i in range(10):        
            numExamples = numpy.random.randint(1, 200)
            numFeatures = numpy.random.randint(1, 20)
            minSplit = numpy.random.randint(1, 50)
            maxDepth = numpy.random.randint(0, 10)
            
            X, y = generator.generateBinaryExamples(numExamples, numFeatures)   
            y = numpy.array(y, numpy.float)
                
            learner = DecisionTreeLearner(minSplit=minSplit, maxDepth=maxDepth) 
            learner.learnModel(X, y)    
            
            predY = learner.predict(X)
            
            tree = learner.tree            
            
            for vertexId in tree.getAllVertexIds(): 
                
                nptst.assert_array_equal(tree.getVertex(vertexId).getTrainInds(), tree.getVertex(vertexId).getTestInds())
                
            #Compare against sklearn tree  
            regressor = DecisionTreeRegressor(min_samples_split=minSplit, max_depth=maxDepth, min_density=0.0)
            regressor.fit(X, y)
            
            sktree = regressor.tree_
            
            #Note that the sklearn algorithm appears to combine nodes with same value 
            #self.assertEquals(sktree.node_count, tree.getNumVertices())
            self.assertEquals(sktree.feature[0], tree.getRoot().getFeatureInd())
            self.assertEquals(sktree.value[0], tree.getRoot().getValue())
            self.assertAlmostEquals(sktree.threshold[0], tree.getRoot().getThreshold(), 3)
            
            predY2 = regressor.predict(X)
            
            #Note that this is not always precise because if two thresholds give the same error we choose the largest 
            #and not sure how it is chosen in sklearn (or if the code is correct)
            self.assertTrue(abs(numpy.linalg.norm(predY-y)- numpy.linalg.norm(predY2-y))/numExamples < 0.05)  

    def testRecursiveSetPrune(self): 
        numExamples = 1000
        X, y = data.make_regression(numExamples)  
        
        y = Standardiser().normaliseArray(y)
        
        numTrain = numpy.round(numExamples * 0.66)     
        
        trainX = X[0:numTrain, :]
        trainY = y[0:numTrain]
        testX = X[numTrain:, :]
        testY = y[numTrain:]
        
        learner = DecisionTreeLearner()
        learner.learnModel(trainX, trainY)
        
        rootId = (0,)
        learner.tree.getVertex(rootId).setTestInds(numpy.arange(testX.shape[0]))
        learner.recursiveSetPrune(testX, testY, rootId)
        
        for vertexId in learner.tree.getAllVertexIds(): 
            tempY = testY[learner.tree.getVertex(vertexId).getTestInds()]
            predY = numpy.ones(tempY.shape[0])*learner.tree.getVertex(vertexId).getValue()
            error = numpy.sum((tempY-predY)**2)
            self.assertAlmostEquals(error, learner.tree.getVertex(vertexId).getTestError())
            
        #Check leaf indices form all indices 
        inds = numpy.array([])        
        
        for vertexId in learner.tree.leaves(): 
            inds = numpy.union1d(inds, learner.tree.getVertex(vertexId).getTestInds())
            
        nptst.assert_array_equal(inds, numpy.arange(testY.shape[0]))
        
        
        
    def testprune(self): 
        learner = DecisionTreeLearner(minSplit=5)
        learner.learnModel(self.X, self.y)
        
        unprunedTree = learner.getTree().copy()
        
        learner.cartPrune(self.X, self.y)
    
        self.assertTrue(learner.tree.isSubtree(unprunedTree))
        
    
    def testCvPrune(self): 
        numExamples = 500
        X, y = data.make_regression(numExamples)  
        
        y = Standardiser().standardiseArray(y)
        
        numTrain = numpy.round(numExamples * 0.33)     
        numValid = numpy.round(numExamples * 0.33) 
        
        trainX = X[0:numTrain, :]
        trainY = y[0:numTrain]
        validX = X[numTrain:numTrain+numValid, :]
        validY = y[numTrain:numTrain+numValid]
        testX = X[numTrain+numValid:, :]
        testY = y[numTrain+numValid:]
        
        learner = DecisionTreeLearner()
        learner.learnModel(trainX, trainY)
        error1 = Evaluator.rootMeanSqError(learner.predict(testX), testY)
        
        #print(learner.getTree())
        unprunedTree = learner.tree.copy() 
        learner.setGamma(1000)
        learner.cvPrune(trainX, trainY)
        
        self.assertEquals(unprunedTree.getNumVertices(), learner.tree.getNumVertices())
        learner.setGamma(100)
        learner.cvPrune(trainX, trainY)
        
        #Test if pruned tree is subtree of current: 
        for vertexId in learner.tree.getAllVertexIds(): 
            self.assertTrue(vertexId in unprunedTree.getAllVertexIds())
            
        #The error should be better after pruning 
        learner.learnModel(trainX, trainY)
        #learner.cvPrune(validX, validY, 0.0, 5)
        learner.repPrune(validX, validY)
      
        error2 = Evaluator.rootMeanSqError(learner.predict(testX), testY)
        
        self.assertTrue(error1 >= error2)

    @unittest.skip("")  
    def testModelSelect(self): 
        
        """
        We test the results on some data and compare to SVR. 
        """
        numExamples = 200
        X, y = data.make_regression(numExamples, noise=0.5)  
        
        X = Standardiser().standardiseArray(X)
        y = Standardiser().standardiseArray(y)
        
        trainX = X[0:100, :]
        trainY = y[0:100]
        testX = X[100:, :]
        testY = y[100:]
        
        learner = DecisionTreeLearner(maxDepth=20, minSplit=10, pruneType="REP-CV")
        learner.setPruneCV(8)
        
        paramDict = {} 
        paramDict["setGamma"] = numpy.linspace(0.0, 1.0, 10) 
        paramDict["setPruneCV"] = numpy.arange(6, 11, 2, numpy.int)
        
        folds = 5
        idx = Sampling.crossValidation(folds, trainX.shape[0])
        bestTree, cvGrid = learner.parallelModelSelect(trainX, trainY, idx, paramDict)


        predY = bestTree.predict(testX)
        error = Evaluator.rootMeanSqError(testY, predY)
        print(error)
        
        
        learner = DecisionTreeLearner(maxDepth=20, minSplit=5, pruneType="CART")
        
        paramDict = {} 
        paramDict["setGamma"] = numpy.linspace(0.0, 1.0, 50) 
        
        folds = 5
        idx = Sampling.crossValidation(folds, trainX.shape[0])
        bestTree, cvGrid = learner.parallelModelSelect(trainX, trainY, idx, paramDict)


        predY = bestTree.predict(testX)
        error = Evaluator.rootMeanSqError(testY, predY)
        print(error)
              
        return 
        #Let's compare to the SVM 
        learner2 = LibSVM(kernel='gaussian', type="Epsilon_SVR") 
        
        paramDict = {} 
        paramDict["setC"] = 2.0**numpy.arange(-10, 14, 2, dtype=numpy.float)
        paramDict["setGamma"] = 2.0**numpy.arange(-10, 4, 2, dtype=numpy.float)
        paramDict["setEpsilon"] = learner2.getEpsilons()
        
        idx = Sampling.crossValidation(folds, trainX.shape[0])
        bestSVM, cvGrid = learner2.parallelModelSelect(trainX, trainY, idx, paramDict)

        predY = bestSVM.predict(testX)
        error = Evaluator.rootMeanSqError(testY, predY)
        print(error)

    def testCARTPrune(self): 
        numExamples = 500
        X, y = data.make_regression(numExamples)  
        
        y = Standardiser().standardiseArray(y)
        
        numTrain = numpy.round(numExamples * 0.33)     
        numValid = numpy.round(numExamples * 0.33) 
        
        trainX = X[0:numTrain, :]
        trainY = y[0:numTrain]
        validX = X[numTrain:numTrain+numValid, :]
        validY = y[numTrain:numTrain+numValid]
        testX = X[numTrain+numValid:, :]
        testY = y[numTrain+numValid:]
        
        learner = DecisionTreeLearner(pruneType="none", maxDepth=10, minSplit=2)
        learner.learnModel(trainX, trainY)    
        
        learner = DecisionTreeLearner(pruneType="CART", maxDepth=10, minSplit=2, gamma=1000)
        learner.learnModel(trainX, trainY)
        self.assertTrue(learner.tree.getNumVertices() <= 1000)
        predY = learner.predict(trainX)

        learner.setGamma(200)
        learner.learnModel(trainX, trainY)
        self.assertTrue(learner.tree.getNumVertices() <= 200)
        
        learner.setGamma(100)
        learner.learnModel(trainX, trainY)
        self.assertTrue(learner.tree.getNumVertices() <= 100)
        

        learner = DecisionTreeLearner(pruneType="none", maxDepth=10, minSplit=2)
        learner.learnModel(trainX, trainY)
        predY2 = learner.predict(trainX)
        
        #Gamma = 0 implies no pruning 
        nptst.assert_array_equal(predY, predY2)
        
        #Full pruning 
        learner = DecisionTreeLearner(pruneType="CART", maxDepth=3, gamma=1)
        learner.learnModel(trainX, trainY)
        self.assertEquals(learner.tree.getNumVertices(), 1)
        
    def testParallelPen(self): 
        #Check if penalisation == inf when treeSize < gamma 
        numExamples = 100
        X, y = data.make_regression(numExamples) 
        learner = DecisionTreeLearner(pruneType="CART", maxDepth=10, minSplit=2)
        
        paramDict = {} 
        paramDict["setGamma"] = numpy.array(numpy.round(2**numpy.arange(1, 10, 0.5)-1), dtype=numpy.int)
        
        folds = 3
        alpha = 1.0
        Cvs = numpy.array([(folds-1)*alpha])
        
        idx = Sampling.crossValidation(folds, X.shape[0])
        
        resultsList = learner.parallelPen(X, y, idx, paramDict, Cvs)
        
        learner, trainErrors, currentPenalties = resultsList[0]
        
        learner.setGamma(2**10)
        treeSize = 0
        #Let's work out the size of the unpruned tree 
        for trainInds, testInds in idx: 
            trainX = X[trainInds, :]
            trainY = y[trainInds]
            
            learner.learnModel(trainX, trainY)
            treeSize += learner.tree.size 
        
        treeSize /= float(folds)         
        
        self.assertTrue(numpy.isinf(currentPenalties[paramDict["setGamma"]>treeSize]).all())      
        self.assertTrue(not numpy.isinf(currentPenalties[paramDict["setGamma"]<treeSize]).all())

    def testLearningRate(self): 
        numExamples = 100
        trainX, trainY = data.make_regression(numExamples) 
        trainX = Standardiser().normaliseArray(trainX)
        trainY = Standardiser().normaliseArray(trainY)
        learner = DecisionTreeLearner(pruneType="CART", maxDepth=20, minSplit=1)
        
        
        foldsSet = numpy.arange(2, 7, 2)
        
        gammas = numpy.array(numpy.round(2**numpy.arange(1, 8, 1)-1), dtype=numpy.int)
        paramDict = {} 
        paramDict["setGamma"] = gammas
        
        betaGrid = learner.learningRate(trainX, trainY, foldsSet, paramDict)
        
        #Compute beta more directly 
        numParams = gammas.shape[0]
        sampleSize = trainX.shape[0]
        sampleMethod = Sampling.crossValidation
        Cvs = numpy.array([1])
        penalties = numpy.zeros((foldsSet.shape[0], numParams))
        betas = numpy.zeros(gammas.shape[0])
        
        for k in range(foldsSet.shape[0]): 
            folds = foldsSet[k]
            logging.debug("Folds " + str(folds))
            
            idx = sampleMethod(folds, trainX.shape[0])   
            
            #Now try penalisation
            resultsList = learner.parallelPen(trainX, trainY, idx, paramDict, Cvs)
            bestLearner, trainErrors, currentPenalties = resultsList[0]
            penalties[k, :] = currentPenalties
        
        for i in range(gammas.shape[0]): 
            inds = numpy.logical_and(numpy.isfinite(penalties[:, i]), penalties[:, i]>0)
            tempPenalties = penalties[:, i][inds]
            tempfoldsSet = numpy.array(foldsSet, numpy.float)[inds]                            
            
            if tempPenalties.shape[0] > 1: 
                x = numpy.log((tempfoldsSet-1)/tempfoldsSet*sampleSize)
                y = numpy.log(tempPenalties)+numpy.log(tempfoldsSet)   
            
                clf = linear_model.LinearRegression()
                clf.fit(numpy.array([x]).T, y)
                betas[i] = clf.coef_[0]    
                
        betas = -betas   
        
        nptst.assert_array_equal(betaGrid, betas)
        
if __name__ == "__main__":
    unittest.main()