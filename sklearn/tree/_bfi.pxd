# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Mikail Yayla <mikail.yayla@tu-dortmund.de>
#
# License: BSD 3 clause
from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.string cimport memcpy

import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(np.ndarray arr, PyObject* obj)

# Types
ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer
ctypedef np.npy_uint64 UINT64_t          # Unsigned 64 bit integer

# bit flip injection into floating point value (32 bit)
cdef DTYPE_t bfi_float (DTYPE_t x, DTYPE_t ber)
# bit flip injection into integer value
cdef SIZE_t bfi_intp (SIZE_t x, DTYPE_t ber, SIZE_t bits)
