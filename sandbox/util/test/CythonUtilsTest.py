import unittest
import numpy
import numpy.testing as nptst
from sandbox.util.CythonUtils import inverseChoicePy, choicePy, dotPy

class  CythonUtilsTest(unittest.TestCase):
    def testInverseChoicePy(self):
        n = 100
        a = numpy.array(numpy.random.randint(0, n, 50), numpy.int32)
        a = numpy.unique(a)

        numRuns = 1000 
        for i in range(numRuns): 
            j = inverseChoicePy(a, n)
            self.assertTrue(j not in a)
        

    def testChoicePy(self): 
        n = 100
        k = 50
        a = numpy.array(numpy.random.randint(0, n, k), numpy.int32)
        a = numpy.unique(a)
        probs = numpy.ones(a.shape[0])/float(a.shape[0])
        
        sample = choicePy(a, 10, probs)

        for item in sample:
            self.assertTrue(item in a)

        probs = numpy.zeros(a.shape[0])
        probs[2] = 1
        sample = choicePy(a, 10, probs)
        
        for item in sample:
            self.assertEquals(item, a[2])
            
        a = numpy.array([0, 1, 2], numpy.int32)
        probs = numpy.array([0.2, 0.6, 0.2])
        cumProbs = numpy.cumsum(probs)
        
        runs = 10000
        sample = choicePy(a, runs, cumProbs)
        
        nptst.assert_array_almost_equal(numpy.bincount(sample)/float(runs), probs, 2)
        
    def testDotPy(self): 
        m = 10 
        n = 20 
        k = 5 

        U = numpy.random.rand(m, k)
        V = numpy.random.rand(n, k)

        for i in range(m): 
            for j in range(n): 
                self.assertAlmostEquals(U[i, :].dot(V[j, :]), dotPy(U, i, V, j, k))
        
if __name__ == '__main__':
    unittest.main()