


import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging
import scipy.sparse 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.util.Util import Util 
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.recommendation.SoftImpute import SoftImpute 

class SparseUtilsCythonTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)

    def testGenerateSparseLowRank(self): 
        shape = (5000, 1000)
        r = 5 
        k = 10 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)         
        
        self.assertEquals(U.shape, (shape[0],r))
        self.assertEquals(V.shape, (shape[1], r))
        self.assertTrue(X.nnz <= k)
        
        Y = (U*s).dot(V.T)
        inds = X.nonzero()
        
        for i in range(inds[0].shape[0]):
            self.assertAlmostEquals(X[inds[0][i], inds[1][i]], Y[inds[0][i], inds[1][i]])
 
    def testGenerateLowRank(self): 
        shape = (5000, 1000)
        r = 5  
        
        U, s, V = SparseUtils.generateLowRank(shape, r)
        
        nptst.assert_array_almost_equal(U.T.dot(U), numpy.eye(r))
        nptst.assert_array_almost_equal(V.T.dot(V), numpy.eye(r))
        
        self.assertEquals(U.shape[0], shape[0])
        self.assertEquals(V.shape[0], shape[1])
        self.assertEquals(s.shape[0], r)
        
        #Check the range is not 
        shape = (500, 500)
        r = 100
        U, s, V = SparseUtils.generateLowRank(shape, r)
        X = (U*s).dot(V.T)
        
        self.assertTrue(abs(numpy.max(X) - 1) < 0.5) 
        self.assertTrue(abs(numpy.min(X) + 1) < 0.5) 
       

    def testReconstructLowRank(self): 
        shape = (5000, 1000)
        r = 5
        
        U, s, V = SparseUtils.generateLowRank(shape, r)
        
        inds = numpy.array([0])
        X = SparseUtils.reconstructLowRank(U, s, V, inds)
        
        self.assertAlmostEquals(X[0, 0], (U[0, :]*s).dot(V[0, :]))
        
    def testSvdSoft(self): 
        A = scipy.sparse.rand(10, 10, 0.2)
        A = A.tocsc()
        
        lmbda = 0.2
        U, s, V = SparseUtils.svdSoft(A, lmbda)
        ATilde = U.dot(numpy.diag(s)).dot(V.T)     
        
        #Now compute the same matrix using numpy
        A = A.todense() 
        
        U2, s2, V2 = numpy.linalg.svd(A)
        inds = numpy.flipud(numpy.argsort(s2))
        inds = inds[s2[inds] > lmbda]
        U2, s2, V2 = Util.indSvd(U2, s2, V2, inds) 
        
        s2 = s2 - lmbda 
        s2 = numpy.clip(s, 0, numpy.max(s2)) 

        ATilde2 = U2.dot(numpy.diag(s2)).dot(V2.T)
        
        nptst.assert_array_almost_equal(s, s)
        nptst.assert_array_almost_equal(ATilde, ATilde2)
        
        #Now run svdSoft with a numpy array 
        U3, s3, V3 = SparseUtils.svdSoft(A, lmbda)
        ATilde3 = U.dot(numpy.diag(s)).dot(V.T)  
        
        nptst.assert_array_almost_equal(s, s3)
        nptst.assert_array_almost_equal(ATilde3, ATilde2)

    def testSvdSparseLowRank(self): 
        numRuns = 10   
        n = 10
        density = 0.2

        for i in range(numRuns):    
            
            A = scipy.sparse.rand(n, n, density) 
            A = A.tocsc()
            
            B = numpy.random.rand(n, n)
            U, s, V = numpy.linalg.svd(B)
            V = V.T         
            
            r = numpy.random.randint(2, n)
            U = U[:, 0:r]
            s = s[0:r]
            V = V[:, 0:r]
            #B is low rank 
            B = (U*s).dot(V.T)
            
            k = numpy.random.randint(1, r)
            U2, s2, V2 = SparseUtils.svdSparseLowRank(A, U, s, V)
            U2 = U2[:, 0:k]
            s2 = s2[0:k]
            V2 = V2[:, 0:k]
                        
            nptst.assert_array_almost_equal(U2.T.dot(U2), numpy.eye(U2.shape[1]))
            nptst.assert_array_almost_equal(V2.T.dot(V2), numpy.eye(V2.shape[1]))
            #self.assertEquals(s2.shape[0], r)
            
            A2 = (U2*s2).dot(V2.T)
            
            #Compute real SVD 
            C = numpy.array(A.todense()) + B
            U3, s3, V3 = numpy.linalg.svd(C)
            V3 = V3.T  
            U3 = U3[:, 0:k]
            s3 = s3[0:k]
            V3 = V3[:, 0:k]
    
            A3 = (U3*s3).dot(V3.T)
            
            #self.assertAlmostEquals(numpy.linalg.norm(A2 - A3), 0)
            nptst.assert_array_almost_equal(s2, s3, 3)
            nptst.assert_array_almost_equal(numpy.abs(U2), numpy.abs(U3), 3)
            nptst.assert_array_almost_equal(numpy.abs(V2), numpy.abs(V3), 3)
            
    def testGenerateSparseLowRank2(self): 
        shape = (2000, 1000)
        r = 5 
        k = 20000 

        X, U, V = SparseUtils.generateSparseLowRank2(shape, r, k, verbose=True)         
        
        self.assertEquals(U.shape, (shape[0],r))
        self.assertEquals(V.shape, (shape[1], r))
        self.assertTrue(X.nnz <= k)
        
        Y = U.dot(V.T)
        inds = X.nonzero()
        
    def testSvdPropack(self): 
        shape = (500, 100)
        r = 5 
        k = 1000 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)                
        
        k2 = 10 
        U, s, V = SparseUtils.svdPropack(X, k2)

        U2, s2, V2 = numpy.linalg.svd(X.todense())
        V2 = V2.T

        nptst.assert_array_almost_equal(s, s2[0:k2])
        nptst.assert_array_almost_equal(numpy.abs(U), numpy.abs(U2[:, 0:k2]), 3)
        nptst.assert_array_almost_equal(numpy.abs(V), numpy.abs(V2[:, 0:k2]), 3)
                
    def testSvdArpack(self): 
        shape = (500, 100)
        r = 5 
        k = 1000 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)                
        
        k2 = 10 
        U, s, V = SparseUtils.svdArpack(X, k2)

        U2, s2, V2 = numpy.linalg.svd(X.todense())
        V2 = V2.T

        nptst.assert_array_almost_equal(s, s2[0:k2])
        nptst.assert_array_almost_equal(numpy.abs(U), numpy.abs(U2[:, 0:k2]), 3)
        nptst.assert_array_almost_equal(numpy.abs(V), numpy.abs(V2[:, 0:k2]), 3)
        
    def testCentreRows(self): 
        shape = (50, 10)
        r = 5 
        k = 100 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)   
        rowInds, colInds = X.nonzero()
        
        for i in range(rowInds.shape[0]): 
            self.assertEquals(X[rowInds[i], colInds[i]], X.data[i])
        
        mu2 = numpy.array(X.sum(1)).ravel()
        numNnz = numpy.zeros(X.shape[0])
        
        for i in range(X.shape[0]): 
            for j in range(X.shape[1]):     
                if X[i,j]!=0:                 
                    numNnz[i] += 1
                    
        mu2 /= numNnz 
        mu2[numNnz==0] = 0
        
        X, mu = SparseUtils.centerRows(X)      
        nptst.assert_array_almost_equal(numpy.array(X.mean(1)).ravel(), numpy.zeros(X.shape[0]))
        nptst.assert_array_almost_equal(mu, mu2)
        
    def testCentreRows2(self): 
        shape = (50, 10)
        r = 5 
        k = 100 
        
        #Test if centering rows changes the RMSE
        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)   
 
        Y = X.copy() 
        Y.data = numpy.random.rand(X.nnz)
        
        error = ((X.data - Y.data)**2).sum()
        
        X, mu = SparseUtils.centerRows(X)
        Y, mu = SparseUtils.centerRows(Y, mu)
        
        error2 = ((X.data - Y.data)**2).sum()
        self.assertAlmostEquals(error, error2)
        
        error3 = numpy.linalg.norm(X.todense()- Y.todense())**2
        self.assertAlmostEquals(error2, error3)        
        
        
    def testCentreCols(self): 
        shape = (50, 10)
        r = 5 
        k = 100 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)   
        rowInds, colInds = X.nonzero()
        
        mu2 = numpy.array(X.sum(0)).ravel()
        numNnz = numpy.zeros(X.shape[1])
        
        for i in range(X.shape[0]): 
            for j in range(X.shape[1]):     
                if X[i,j]!=0:                 
                    numNnz[j] += 1
                    
        mu2 /= numNnz 
        mu2[numNnz==0] = 0
        
        X, mu = SparseUtils.centerCols(X)      
        nptst.assert_array_almost_equal(numpy.array(X.mean(0)).ravel(), numpy.zeros(X.shape[1]))
        nptst.assert_array_almost_equal(mu, mu2)       
        
    def testUncentre(self): 
        shape = (50, 10)
        r = 5 
        k = 100 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)   
        rowInds, colInds = X.nonzero()  
        
        Y = X.copy()

        inds = X.nonzero()
        X, mu1 = SparseUtils.centerRows(X)
        X, mu2 = SparseUtils.centerCols(X, inds=inds)   
        
        cX = X.copy()
        
        Y2 = SparseUtils.uncenter(X, mu1, mu2)
        
        nptst.assert_array_almost_equal(Y.todense(), Y2.todense(), 3)
        
        #We try softImpute on a centered matrix and check if the results are the same 
        lmbdas = numpy.array([0.1])
        softImpute = SoftImpute(lmbdas)
        
        Z = softImpute.learnModel(cX, fullMatrices=False)
        Z = softImpute.predict([Z], cX.nonzero())[0]
        
        error1 = MCEvaluator.rootMeanSqError(cX, Z)
        
        X = SparseUtils.uncenter(cX, mu1, mu2)
        Z2 = SparseUtils.uncenter(Z, mu1, mu2)
        
        error2 = MCEvaluator.rootMeanSqError(X, Z2)
        
        self.assertAlmostEquals(error1, error2)
 
    def testNonzeroRowColProbs(self): 
        m = 10 
        n = 5 
        X = scipy.sparse.rand(m, n, 0.5)
        X = X.tocsc()
        
        u, v = SparseUtils.nonzeroRowColsProbs(X)
        
        self.assertEquals(u.sum(), 1.0)
        self.assertEquals(v.sum(), 1.0)
        
        X  = numpy.diag(numpy.ones(5))
        X = scipy.sparse.csc_matrix(X)
        
        u, v = SparseUtils.nonzeroRowColsProbs(X)
        
        nptst.assert_array_almost_equal(u, numpy.ones(5)/5)
        nptst.assert_array_almost_equal(v, numpy.ones(5)/5)
        self.assertEquals(u.sum(), 1.0)
        self.assertEquals(v.sum(), 1.0)
   
    def testSubmatrix(self): 
        import sppy 
        numRuns = 100 
        
        for i in range(numRuns): 
            m = numpy.random.randint(5, 50)
            n = numpy.random.randint(5, 50)  
            X = scipy.sparse.rand(m, n, 0.5)
            X = X.tocsc()
            
            inds1 = numpy.arange(0, X.nnz/2)
            inds2 = numpy.arange(X.nnz/2, X.nnz)
            
            X1 = SparseUtils.submatrix(X, inds1)
            X2 = SparseUtils.submatrix(X, inds2)
            
            nptst.assert_array_almost_equal((X1+X2).todense(), X.todense()) 
            
        inds = X.nnz
        X1 = SparseUtils.submatrix(X, inds)
        nptst.assert_array_almost_equal((X1).todense(), X.todense()) 
        
        inds = 2
        X1 = SparseUtils.submatrix(X, inds)
        self.assertTrue(X1.nnz, 2)
        
        #Test with sppy 
        for i in range(numRuns): 
            m = numpy.random.randint(5, 50)
            n = numpy.random.randint(5, 50)  
            X = scipy.sparse.rand(m, n, 0.5)
            X = X.tocsc()
            
            X = sppy.csarray(X)
            
            inds1 = numpy.arange(0, X.nnz/2)
            inds2 = numpy.arange(X.nnz/2, X.nnz)
            
            X1 = SparseUtils.submatrix(X, inds1)
            X2 = SparseUtils.submatrix(X, inds2)
            
            nptst.assert_array_almost_equal((X1+X2).toarray(), X.toarray()) 
    
    @unittest.skip("")
    def testPruneMatrix(self): 
        m = 50 
        n = 30 
        density = 0.5 
        X = scipy.sparse.rand(m, n, density)
        X = X.tocsc()
        
        newX = SparseUtils.pruneMatrix(X, 0, 0)
        
        nptst.assert_array_almost_equal(newX.todense(), X.todense())   
        
        X = numpy.array([[0, 0, 0.1, 0.2], [0, 0.5, 0.1, 0.2], [0, 0, 0.0, 0.2], [0, 0, 0.1, 0.2]])
        X = scipy.sparse.csc_matrix(X)
        
        newX = SparseUtils.pruneMatrix(X, 2, 0)
        
        nptst.assert_array_almost_equal(newX.todense(), numpy.array([[0, 0, 0.1, 0.2], [0, 0.5, 0.1, 0.2], [0, 0, 0.1, 0.2]]))
        
        newX = SparseUtils.pruneMatrix(X, 0, 2)
        
        nptst.assert_array_almost_equal(newX.todense(), numpy.array([[0.1, 0.2], [0.1, 0.2], [0.0, 0.2], [0.1, 0.2]]))

    def testHellingerDistances(self): 
        m = 10 
        n = 5 
        density = 0.5 
        numRuns = 10         
        
        for j in range(numRuns): 
            X = scipy.sparse.rand(m, n, density)
            X = X.tocsc()
            
            v = scipy.sparse.rand(1, n, density)
            
            distances = SparseUtils.hellingerDistances(X, v) 
            
            X = numpy.array(X.todense()) 
            v = numpy.array(v.todense())
            
            distances2 = numpy.zeros(X.shape[0])
            
            for i in range(X.shape[0]): 
                distances2[i] = numpy.linalg.norm(numpy.sqrt(X[i, :]) - numpy.sqrt(v)) * numpy.sqrt(0.5)
          
            nptst.assert_array_almost_equal(distances, distances2)    
            
    def testStandardise(self): 
        m = 10 
        n = 5 
        density = 0.5 
        numRuns = 10 
        
        X = scipy.sparse.rand(m, n, density)
        X = X.tocsc()

        X2 = SparseUtils.standardise(X)
        X2.data = X2.data**2
        nptst.assert_array_almost_equal(numpy.array(X2.sum(0)).ravel(), numpy.ones(n)) 
     
    def testGenerateSparseBinaryMatrix(self):
        m = 5 
        n = 10 
        k = 3
        quantile = 0.7
        numpy.random.seed(21)
        X = SparseUtils.generateSparseBinaryMatrix((m,n), k, quantile)
        Xscipy = numpy.array(X.todense()) 
        
        nptst.assert_array_equal(numpy.array(X.sum(1)).flatten(), numpy.ones(m)*3)
        
        quantile = 0.0 
        X = SparseUtils.generateSparseBinaryMatrix((m,n), k, quantile)
        self.assertTrue(numpy.linalg.norm(X - numpy.ones((m,n))) < 1.1)
        #nptst.assert_array_almost_equal(X.todense(), numpy.ones((m,n)))
        
        quantile = 0.7
        numpy.random.seed(21)
        X = SparseUtils.generateSparseBinaryMatrix((m,n), k, quantile, csarray=True)
        Xcsarray = X.toarray()
        
        nptst.assert_array_equal(numpy.array(X.sum(1)).flatten(), numpy.ones(m)*3)
        
        quantile = 0.0 
        X = SparseUtils.generateSparseBinaryMatrix((m,n), k, quantile, csarray=True)
        self.assertTrue(numpy.linalg.norm(X.toarray() - numpy.ones((m,n))) < 1.1)
        #nptst.assert_array_almost_equal(X.toarray(), numpy.ones((m,n)))
        
        nptst.assert_array_equal(Xcsarray, Xscipy)
        
        #Test variation in the quantiles 
        w = 0.7
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, sd=0.1, csarray=True, verbose=True)
        
        Z = (U*s).dot(V.T)
        X2 = numpy.zeros((m, n))
        r2 = numpy.zeros(m)
        for i in range(m): 
            r2[i] = numpy.percentile(numpy.sort(Z[i, :]), wv[i]*100)
            X2[i, Z[i, :]>r2[i]] = 1 
        r = SparseUtilsCython.computeR2(U*s, V, wv)

        nptst.assert_array_almost_equal(X.toarray(), X2)
        nptst.assert_array_almost_equal(r, r2)
        
        #Test a larger standard deviation
        w = 0.7
        X, U, s, V, wv = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, sd=0.5, csarray=True, verbose=True)
        
        Z = (U*s).dot(V.T)
        X2 = numpy.zeros((m, n))
        r2 = numpy.zeros(m)
        for i in range(m): 
            r2[i] = numpy.percentile(numpy.sort(Z[i, :]), wv[i]*100)
            X2[i, Z[i, :]>=r2[i]] = 1 
        r = SparseUtilsCython.computeR2(U*s, V, wv)

        nptst.assert_array_almost_equal(X.toarray(), X2)
        nptst.assert_array_almost_equal(r, r2)
        
    def testEquals(self):
        A = numpy.array([[4, 2, 1], [6, 3, 9], [3, 6, 0]])
        B = numpy.array([[4, 2, 1], [6, 3, 9], [3, 6, 0]])

        A = scipy.sparse.csr_matrix(A)
        B = scipy.sparse.csr_matrix(B)

        self.assertTrue(SparseUtils.equals(A, B))

        A[0, 1] = 5
        self.assertFalse(SparseUtils.equals(A, B))

        A[0, 1] = 2
        B[0, 1] = 5
        self.assertFalse(SparseUtils.equals(A, B))

        A[2, 2] = -1
        self.assertFalse(SparseUtils.equals(A, B))

        #Test two empty graphs
        A = scipy.sparse.csr_matrix((5, 5)) 
        B = scipy.sparse.csr_matrix((5, 5))

        self.assertTrue(SparseUtils.equals(A, B))

    def testNorm(self):
        numRows = 10
        numCols = 10

        for k in range(10):
            A = scipy.sparse.rand(numRows, numCols, 0.1, "csr")

            norm = SparseUtils.norm(A)

            norm2 = 0
            for i in range(numRows):
                for j in range(numCols):
                    norm2 += A[i, j]**2

            norm2 = numpy.sqrt(norm2)
            norm3 = numpy.linalg.norm(numpy.array(A.todense()))
            self.assertAlmostEquals(norm, norm2)
            self.assertAlmostEquals(norm, norm3)

    def testResize(self): 
        numRows = 10
        numCols = 10        
        
        A = scipy.sparse.rand(numRows, numCols, 0.1, "csr") 
        
        B = SparseUtils.resize(A, (5, 5))
        
        self.assertEquals(B.shape, (5, 5))
        for i in range(5): 
            for j in range(5): 
                self.assertEquals(B[i,j], A[i,j])
                
        B = SparseUtils.resize(A, (15, 15))
        
        self.assertEquals(B.shape, (15, 15))
        self.assertEquals(B.nnz, A.nnz) 
        for i in range(10): 
            for j in range(10): 
                self.assertEquals(B[i,j], A[i,j])


    def testDiag(self):
        numRows = 10
        numCols = 10  
        A = scipy.sparse.rand(numRows, numCols, 0.5, "csr")

        d = SparseUtils.diag(A)

        for i in range(numRows): 
            self.assertEquals(d[i], A[i,i])             

    def testSelectMatrix(self): 
        numRows = 10
        numCols = 10  
        A = scipy.sparse.rand(numRows, numCols, 0.5, "csr")
        
        #Select first row 
        rowInds = numpy.zeros(numCols)
        colInds = numpy.arange(10)

        newA = SparseUtils.selectMatrix(A, rowInds, colInds)
        
        for i in range(numCols): 
            self.assertEquals(A[0, i], newA[0, i])
            
        for i in range(1, numRows): 
            for j in range(numCols): 
                self.assertEquals(newA[i, j], 0)
   
    def testSplitNnz(self): 
        numRuns = 100 
        import sppy 
        
        for i in range(numRuns): 
            m = numpy.random.randint(5, 50)
            n = numpy.random.randint(5, 50)  
            X = scipy.sparse.rand(m, n, 0.5)
            X = X.tocsc()
            
            split = numpy.random.rand()
            X1, X2 = SparseUtils.splitNnz(X, split)
            
            nptst.assert_array_almost_equal((X1+X2).todense(), X.todense()) 
            
        for i in range(numRuns): 
            m = numpy.random.randint(5, 50)
            n = numpy.random.randint(5, 50)  
            X = scipy.sparse.rand(m, n, 0.5)
            X = X.tocsc()
            
            X = sppy.csarray(X)
            
            split = numpy.random.rand()
            X1, X2 = SparseUtils.splitNnz(X, split)
            
            nptst.assert_array_almost_equal((X1+X2).toarray(), X.toarray()) 
   
    def testGetOmegaList(self):
        import sppy 
        m = 10 
        n = 5
        X = scipy.sparse.rand(m, n, 0.1)
        X = X.tocsr()
        
        
        omegaList = SparseUtils.getOmegaList(X)
        for i in range(m): 
            nptst.assert_array_almost_equal(omegaList[i], X.toarray()[i, :].nonzero()[0])
        
        Xsppy = sppy.csarray(X)
        omegaList = SparseUtils.getOmegaList(Xsppy)
        
        for i in range(m):
            nptst.assert_array_almost_equal(omegaList[i], X.toarray()[i, :].nonzero()[0])
    
    def testGetOmegaListPtr(self): 
        import sppy 
        m = 10 
        n = 5
        X = scipy.sparse.rand(m, n, 0.1)
        X = X.tocsr()
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)

        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            nptst.assert_array_almost_equal(omegai, X.toarray()[i, :].nonzero()[0])
        
        Xsppy = sppy.csarray(X)
        indPtr, colInds  = SparseUtils.getOmegaListPtr(Xsppy)
        
        for i in range(m):
            omegai = colInds[indPtr[i]:indPtr[i+1]]
            nptst.assert_array_almost_equal(omegai, X.toarray()[i, :].nonzero()[0])
        
        #Test a zero array (scipy doesn't work in this case)
        X = sppy.csarray((m,n))
        
        indPtr, colInds = SparseUtils.getOmegaListPtr(X)
   
        for i in range(m): 
            omegai = colInds[indPtr[i]:indPtr[i+1]]
   
if __name__ == '__main__':
    unittest.main()
