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
# ctypedef np.npy_float32 DTYPE_t          # Type of X
# ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
# ctypedef np.npy_intp SIZE_t              # Type for indices and counters
# ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
# ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# bit flip injection into floating point value (32 bit)
cdef DTYPE_t bfi_float (DTYPE_t x, DTYPE_t ber):
    cdef np.ndarray[DTYPE_t] randomval = np.random.uniform(low=0.0, high=1.0, size=32).astype('float32')
    #print(randomval)

    # This way of type punning is undefined behavior (see strict aliasing rule)
    # cdef UINT32_t* bfi_val_uintp = <unsigned int*> (&x)
    #cdef UINT32_t bfi_val = (bfi_val_uintp)[0]

    # use memcpy to do bit operations on floating point (avoids strict aliasing rule)
    cdef UINT32_t bfi_val;
    memcpy(&bfi_val, &x, 4)
    #print(bfi_val)

    # cdef UINT32_t inct = 0x1
    # for i in range(0,32):
    #   if ((bfi_val & inct) == inct):
    #     printf("1")
    #   else:
    #     printf("0")
    #   inct <<= 1
    # printf("\n\n")

    cdef UINT32_t injector = 0x1
    for i in range(0,32):
      if randomval[i] < ber:
        bfi_val ^= injector
      injector <<= 1

    # inct = 0x1
    # for i in range(0,32):
    #   if ((bfi_val & inct) == inct):
    #     printf("1")
    #   else:
    #     printf("0")
    #   inct <<= 1
    # printf("\n\n")

    # cdef np.ndarray[float] out = np.zeros(3, dtype=np.float)
    #cdef np.ndarray[float] out = np.zeros((3,), dtype=np.float32)

    #since cython does not have dereferencing, use array notation
    #cdef float injected_val = (<float*> bfi_val_uintp)[0]
    #cdef float injected_val = (<float*> &bfi_val)[0]

    memcpy(&x, &bfi_val, 4)

    return x
