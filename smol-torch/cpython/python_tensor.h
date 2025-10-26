#ifndef PYTHON_TENSOR_H
#define PYTHON_TENSOR_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "tensor.h"

typedef struct {
    PyObject_HEAD
    Tensor* tensor;
} PyTensorObject;

extern PyTypeObject PyTensorType;

#endif // PYTHON_TENSOR_H
