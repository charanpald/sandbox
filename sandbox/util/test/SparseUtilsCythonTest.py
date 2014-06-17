

import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging
import scipy.sparse 
from sandbox.util.SparseUtilsCython import SparseUtilsCython

class SparseUtilsCythonTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)

    def testPartialReconstructValsPQ(self):
        n = 10
        Y = numpy.random.rand(n, n)
        
        U, s, V = numpy.linalg.svd(Y)
        V = V.T 
        
        V = numpy.ascontiguousarray(V)
        
        rowInds, colInds = numpy.nonzero(Y)  
        rowInds = numpy.array(rowInds, numpy.int32)
        colInds = numpy.array(colInds, numpy.int32)
        vals = SparseUtilsCython.partialReconstructValsPQ(rowInds, colInds, numpy.ascontiguousarray(U*s), V)
        X = numpy.reshape(vals, Y.shape)
        
        nptst.assert_almost_equal(X, Y)
        
        #Try just some indices 
        density = 0.2
        A = scipy.sparse.rand(n, n, density)
        inds = A.nonzero()
        rowInds = numpy.array(inds[0], numpy.int32)
        colInds = numpy.array(inds[1], numpy.int32)
        
        vals = SparseUtilsCython.partialReconstructValsPQ(rowInds, colInds, numpy.ascontiguousarray(U*s), V)
        
        for i in range(inds[0].shape[0]): 
            j = inds[0][i]
            k = inds[1][i]
            
            self.assertAlmostEquals(vals[i], Y[j, k])  
            
        
        self.assertEquals(A.nnz, inds[0].shape[0])

    def testPartialReconstructValsPQ2(self): 
        numRuns = 10         
        
        for i in range(numRuns): 
            m = numpy.random.randint(5, 50)
            n = numpy.random.randint(5, 50)
            Y = numpy.random.rand(m, n)
            
            U, s, V = numpy.linalg.svd(Y,  full_matrices=0)
            V = V.T 
            
            V = numpy.ascontiguousarray(V)
            
            rowInds, colInds = numpy.nonzero(Y)  
            rowInds = numpy.array(rowInds, numpy.int32)
            colInds = numpy.array(colInds, numpy.int32)
            #print(U.shape, V.shape)
            vals = SparseUtilsCython.partialReconstructValsPQ(rowInds, colInds, numpy.ascontiguousarray(U*s), V)
            X = numpy.reshape(vals, Y.shape)
            
            nptst.assert_almost_equal(X, Y)
        

    def testPartialOuterProduct(self):
        m = 15        
        n = 10
        
        
        u = numpy.random.rand(m)
        v = numpy.random.rand(n)
        Y = numpy.outer(u, v)
        
        inds = numpy.nonzero(Y)
        rowInds = numpy.array(inds[0], numpy.int32)
        colInds = numpy.array(inds[1], numpy.int32)
        vals = SparseUtilsCython.partialOuterProduct(rowInds, colInds, u, v)
        X = numpy.reshape(vals, Y.shape)
        
        nptst.assert_almost_equal(X, Y)
        
        #Try just some indices 
        density = 0.2
        A = scipy.sparse.rand(n, n, density)
        inds = A.nonzero()
        rowInds = numpy.array(inds[0], numpy.int32)
        colInds = numpy.array(inds[1], numpy.int32)
        
        vals = SparseUtilsCython.partialOuterProduct(rowInds, colInds, u, v)
        
        for i in range(inds[0].shape[0]): 
            j = inds[0][i]
            k = inds[1][i]
            
            self.assertAlmostEquals(vals[i], Y[j, k])  
            
        
        self.assertEquals(A.nnz, inds[0].shape[0])

    def testSumCols(self): 
        A = scipy.sparse.rand(10, 15, 0.5)*10
        A = scipy.sparse.csc_matrix(A, dtype=numpy.uint8)
        
        rowInds, colInds = A.nonzero()  
        rowInds = numpy.array(rowInds, numpy.int32)
        colInds = numpy.array(colInds, numpy.int32)
        
        sumCol = SparseUtilsCython.sumCols(rowInds, numpy.array(A[rowInds, colInds]).flatten(), A.shape[0])
        nptst.assert_array_equal(numpy.array(A.sum(1)).flatten(), sumCol) 

    def testComputeR(self): 
        U = numpy.random.rand(10, 5)
        V = numpy.random.rand(15, 5)
        
        Z = U.dot(V.T)
        
        u = 1.0
        r = SparseUtilsCython.computeR(U, V, u, indsPerRow=1000)
               
        tol = 0.1
        self.assertTrue(numpy.linalg.norm(Z.max(1) - r)/numpy.linalg.norm(Z.max(1)) < tol)
        
        u = 0.0
        r = SparseUtilsCython.computeR(U, V, u, indsPerRow=1000)
        self.assertTrue(numpy.linalg.norm(Z.min(1) - r)/numpy.linalg.norm(Z.min(1)) < tol)
        
        u = 0.3
        r = SparseUtilsCython.computeR(U, V, u, indsPerRow=1000) 
        r2 = numpy.percentile(Z, u*100.0, 1)
        #nptst.assert_array_almost_equal(r, r2, 2)
        self.assertTrue(numpy.linalg.norm(r - r2)/numpy.linalg.norm(r) < tol)
        
        #Try a larger matrix 
        U = numpy.random.rand(100, 5)
        V = numpy.random.rand(105, 5)
        
        Z = U.dot(V.T)
        
        r = SparseUtilsCython.computeR(U, V, u) 
        r2 = numpy.percentile(Z, u*100.0, 1)
        
        self.assertTrue(numpy.linalg.norm(r-r2) < 0.5)

    def testComputeR2(self): 
        m = 10 
        n = 15
        U = numpy.random.rand(m, 5)
        V = numpy.random.rand(n, 5)
        
        Z = U.dot(V.T)
        
        w = numpy.ones(m)*1.0
        r = SparseUtilsCython.computeR2(U, V, w, indsPerRow=1000)
               
        tol = 0.1
        self.assertTrue(numpy.linalg.norm(Z.max(1) - r)/numpy.linalg.norm(Z.max(1)) < tol)
        
        w =  numpy.zeros(m)
        r = SparseUtilsCython.computeR2(U, V, w, indsPerRow=1000)
        self.assertTrue(numpy.linalg.norm(Z.min(1) - r)/numpy.linalg.norm(Z.min(1)) < tol)
        
        w = numpy.zeros(m)
        w[5:10] = 1
        r = SparseUtilsCython.computeR2(U, V, w, indsPerRow=1000)
        self.assertTrue(numpy.linalg.norm(Z[0:5, :].min(1) - r[0:5])/numpy.linalg.norm(Z[0:5, :].min(1)) < tol)
        self.assertTrue(numpy.linalg.norm(Z[5:, :].max(1) - r[5:])/numpy.linalg.norm(Z[5:, :].min(1)) < tol)
        
        w =  numpy.ones(m)*0.3
        r = SparseUtilsCython.computeR2(U, V, w, indsPerRow=1000) 
        r2 = numpy.zeros(m)
        for i in range(m): 
            r2[i] = numpy.percentile(Z[i, :], w[i]*100.0)
        self.assertTrue(numpy.linalg.norm(r2 - r)/numpy.linalg.norm(r2) < tol)
        
        w =  numpy.random.rand(m)
        r = SparseUtilsCython.computeR2(U, V, w) 
        r2 = numpy.zeros(m)

        for i in range(m): 
            r2[i] = numpy.percentile(Z[i, :], w[i]*100.0)
        self.assertTrue(numpy.linalg.norm(r2 - r)/numpy.linalg.norm(r2) < tol)       
        
        #Try a larger matrix 
        m = 100 
        n = 105
        U = numpy.random.rand(m, 5)
        V = numpy.random.rand(n, 5)
        
        Z = U.dot(V.T)
        w =  numpy.random.rand(m)
        r = SparseUtilsCython.computeR2(U, V, w, indsPerRow=10000) 
        r2 = numpy.zeros(m) 
        for i in range(m): 
            r2[i] = numpy.percentile(Z[i, :], w[i]*100.0)
        
        self.assertTrue(numpy.linalg.norm(r-r2) < 0.4)

if __name__ == '__main__':
    unittest.main()