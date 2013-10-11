
import unittest
import numpy
import logging
import sys
from exp.metabolomics.leafrank.LinearSvmFGs import LinearSvmFGs

class LinearSvmFGsTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.random.seed(21)
        numFeatures = 200
        numExamples = 200

        self.X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        self.y = numpy.sign(self.X.dot(c) < numpy.mean(self.X.dot(c)))

    def testGenerate(self):
        generate = LinearSvmFGs.generate()

        self.X[:, 15:25] = self.X[:, 15:25]*100

        svc = generate()
        svc.setWaveletInds(numpy.arange(100))
        svc.learnModel(self.X, self.y)
        self.assertEquals(numpy.intersect1d(numpy.arange(15,25), svc.getFeatureInds()).shape[0], 10)
        
        predY = svc.predict(self.X)

        #Now test when all features are wavelets
        svc = generate()
        svc.learnModel(self.X, self.y)
        self.assertEquals(numpy.intersect1d(numpy.arange(15,25), svc.getFeatureInds()).shape[0], 10)

        predY = svc.predict(self.X)

    def testSetWeight(self):
        learner = LinearSvmFGs()
        learner.setWeight(0.8)

if __name__ == '__main__':
    unittest.main()

