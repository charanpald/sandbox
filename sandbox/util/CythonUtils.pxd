
import cython
cimport numpy
import numpy

cdef double square(double d)
cdef double dot(numpy.ndarray[double, ndim = 2, mode="c"] U, unsigned int i, numpy.ndarray[double, ndim = 2, mode="c"] V, unsigned int j, unsigned int k)
cdef numpy.ndarray[double, ndim = 1, mode="c"] scale(numpy.ndarray[double, ndim = 2, mode="c"] U, unsigned int i, double d, unsigned int k)
cdef numpy.ndarray[double, ndim = 1, mode="c"] plusEquals(numpy.ndarray[double, ndim = 2, mode="c"] U, unsigned int i, numpy.ndarray[double, ndim = 1, mode="c"] d, unsigned int k)
cdef unsigned int inverseChoice(numpy.ndarray[int, ndim=1, mode="c"] v, unsigned int n)
cdef numpy.ndarray[int, ndim=1, mode="c"] choice(numpy.ndarray[int, ndim=1, mode="c"] inds, unsigned int numSamples, numpy.ndarray[double, ndim=1, mode="c"] cumProbs)
cdef numpy.ndarray[int, ndim=1, mode="c"] uniformChoice(numpy.ndarray[int, ndim=1, mode="c"] inds, unsigned int numSamples)
cdef double partialSum(numpy.ndarray[double, ndim=1, mode="c"] v, numpy.ndarray[int, ndim=1, mode="c"] inds)