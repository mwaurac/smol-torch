#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"

typedef struct {
    PyObject_HEAD
    Tensor* tensor;
} PyTensorObject;

static void PyTensor_dealloc(PyTensorObject* self) {
    if (self->tensor)
        tensor_free(self->tensor);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyTensor_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyTensorObject* self = (PyTensorObject*)type->tp_alloc(type, 0);
    if (self) {
        self->tensor = NULL;
    }
    return (PyObject*)self;
}

PyDoc_STRVAR(PyTensor_init__doc__,
"Tensor(data=None, shape, dtype='float32')\n"
"--\n\n"
"Create a new tensor with the given shape and optional data.\n"
"\n"
"Parameters\n"
"----------\n"
"data : list, optional\n"
"    The data to initialize the tensor (must match shape size).\n"
"shape : list[int]\n"
"    The dimensions of the tensor.\n"
"dtype : str, optional\n"
"    The data type ('float32', 'float64', 'int32', 'int64').\n"
"\n"
"Examples\n"
"--------\n"
">>> import smol_torch\n"
">>> t = smol_torch.Tensor(shape=[2, 3])\n"
">>> t.shape()\n"
"(2, 3)\n"
">>> t2 = smol_torch.Tensor(data=[1, 2, 3], shape=[3])\n"
">>> t2.shape()\n"
"(3,)\n");

static int PyTensor_init(PyTensorObject* self, PyObject* args, PyObject* kwds) {
    PyObject* data_list = NULL;
    PyObject* shape_list = NULL;
    const char* dtype_str = "float32";

    static char* keywords[] = {"data", "shape", "dtype", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOs", keywords, &data_list, &shape_list, &dtype_str)) {
        return -1;
    }

    if (!shape_list) {
        PyErr_SetString(PyExc_TypeError, "shape argument is required");
        return -1;
    }

    // Parse dtype
    Dtype dtype = DTYPE_FLOAT32;
    if (strcmp(dtype_str, "float32") == 0) {
        dtype = DTYPE_FLOAT32;
    } else if (strcmp(dtype_str, "float64") == 0) {
        dtype = DTYPE_FLOAT64;
    } else if (strcmp(dtype_str, "int32") == 0) {
        dtype = DTYPE_INT32;
    } else if (strcmp(dtype_str, "int64") == 0) {
        dtype = DTYPE_INT64;
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        return -1;
    }

    // Parse shape
    if (!PyList_Check(shape_list)) {
        PyErr_SetString(PyExc_TypeError, "Shape must be a list of integers");
        return -1;
    }

    Py_ssize_t ndim = PyList_Size(shape_list);
    if (ndim <= 0 || ndim > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "Invalid number of dimensions");
        return -1;
    }

    int64_t* shape = malloc(ndim * sizeof(int64_t));
    if (!shape) {
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed");
        return -1;
    }

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PyList_GetItem(shape_list, i);
        if (!PyLong_Check(item)) {
            free(shape);
            PyErr_SetString(PyExc_TypeError, "Shape elements must be integers");
            return -1;
        }
        shape[i] = PyLong_AsLongLong(item);
        if (shape[i] <= 0) {
            free(shape);
            PyErr_SetString(PyExc_ValueError, "Shape dimensions must be positive");
            return -1;
        }
    }

    // Handle data
    if (data_list && data_list != Py_None) {
        if (!PyList_Check(data_list)) {
            free(shape);
            PyErr_SetString(PyExc_TypeError, "Data must be a list");
            return -1;
        }

        int64_t expected_size = get_tensor_size(shape, (int32_t)ndim);
        Py_ssize_t data_size = PyList_Size(data_list);
        if (data_size != expected_size) {
            free(shape);
            PyErr_Format(PyExc_ValueError, "Data length (%zd) does not match tensor size (%lld)",
                         data_size, (long long)expected_size);
            return -1;
        }

        size_t dtype_size = get_tensor_dtype_size(dtype);
        void* data_buffer = malloc(dtype_size * expected_size);
        if (!data_buffer) {
            free(shape);
            PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed");
            return -1;
        }

        for (Py_ssize_t i = 0; i < data_size; i++) {
            PyObject* item = PyList_GetItem(data_list, i);
            if (dtype == DTYPE_FLOAT32) {
                if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                    free(shape);
                    free(data_buffer);
                    PyErr_SetString(PyExc_TypeError, "Data elements must be float or int for float32 dtype");
                    return -1;
                }
                ((float*)data_buffer)[i] = (float)PyFloat_AsDouble(item);
            } else if (dtype == DTYPE_FLOAT64) {
                if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                    free(shape);
                    free(data_buffer);
                    PyErr_SetString(PyExc_TypeError, "Data elements must be float or int for float64 dtype");
                    return -1;
                }
                ((double*)data_buffer)[i] = PyFloat_AsDouble(item);
            } else if (dtype == DTYPE_INT32) {
                if (!PyLong_Check(item)) {
                    free(shape);
                    free(data_buffer);
                    PyErr_SetString(PyExc_TypeError, "Data elements must be int for int32 dtype");
                    return -1;
                }
                ((int32_t*)data_buffer)[i] = (int32_t)PyLong_AsLong(item);
            } else if (dtype == DTYPE_INT64) {
                if (!PyLong_Check(item)) {
                    free(shape);
                    free(data_buffer);
                    PyErr_SetString(PyExc_TypeError, "Data elements must be int for int64 dtype");
                    return -1;
                }
                ((int64_t*)data_buffer)[i] = PyLong_AsLongLong(item);
            }
        }

        self->tensor = create_tensor_with_data(data_buffer, shape, (int32_t)ndim, dtype);
        free(shape);
        free(data_buffer);
    } else {
        self->tensor = create_tensor(shape, (int32_t)ndim, dtype);
        free(shape);
    }

    if (!self->tensor) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create tensor");
        return -1;
    }

    return 0;
}

PyDoc_STRVAR(PyTensor_shape__doc__,
"shape(self)\n"
"--\n\n"
"Return the shape of the tensor as a tuple of integers.\n"
"\n"
"Examples\n"
"--------\n"
">>> import smol_torch\n"
">>> t = smol_torch.Tensor(shape=[2, 3])\n"
">>> t.shape()\n"
"(2, 3)\n");

static PyObject* PyTensor_shape(PyTensorObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->tensor) {
        Py_RETURN_NONE;
    }

    PyObject* shape_tuple = PyTuple_New(self->tensor->ndim);
    for (int32_t i = 0; i < self->tensor->ndim; i++) {
        PyTuple_SetItem(shape_tuple, i, PyLong_FromLongLong(self->tensor->shape[i]));
    }
    return shape_tuple;
}

static PyObject* PyTensor_repr(PyTensorObject* self) {
    // TODO: PRINT DATA AND DEVICE
    if (!self->tensor) {
        return PyUnicode_FromString("Tensor([])");
    }

    void* data = self->tensor->data;

    int32_t ndim = self->tensor->ndim;
    PyObject* shape_tuple = PyTensor_shape(self, NULL);
    PyObject* repr = PyUnicode_FromFormat("Tensor(shape=%S, dtype=%s)",
                                         shape_tuple,
                                         self->tensor->dtype == DTYPE_FLOAT32 ? "float32" :
                                         self->tensor->dtype == DTYPE_FLOAT64 ? "float64" :
                                         self->tensor->dtype == DTYPE_INT32 ? "int32" : "int64");
    Py_DECREF(shape_tuple);
    return repr;
}
static PyMemberDef PyTensor_members[] = {
    {NULL}  // Sentinel
};

static PyMethodDef PyTensor_methods[] = {
    {"shape", (PyCFunction)PyTensor_shape, METH_NOARGS, PyTensor_shape__doc__},
    {NULL}  // Sentinel
};

PyDoc_STRVAR(PyTensor__doc__,
"Tensor(data=None, shape, dtype='float32')\n"
"--\n\n"
"A lightweight tensor object.\n"
"\n"
"Examples\n"
"-------\n"
">>> import smol_torch\n"
">>> t = smol_torch.Tensor(shape=[2, 3])\n"
">>> t.shape()\n"
"(2, 3)\n"
">>> t2 = smol_torch.Tensor(data=[1, 2, 3], shape=[3])\n"
">>> t2.shape()\n"
"(3,)\n");

static PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "smol_torch.Tensor",
    .tp_doc = PyTensor__doc__,
    .tp_basicsize = sizeof(PyTensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyTensor_new,
    .tp_init = (initproc)PyTensor_init,
    .tp_dealloc = (destructor)PyTensor_dealloc,
    .tp_repr = (reprfunc)PyTensor_repr,
    .tp_members = PyTensor_members,
    .tp_methods = PyTensor_methods,
};

static struct PyModuleDef smol_torch_module = {
    PyModuleDef_HEAD_INIT,
    "smol_torch",
    "A small torch-like library",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_smol_torch(void) {
    PyObject* module = PyModule_Create(&smol_torch_module);
    if (!module) return NULL;

    if (PyType_Ready(&PyTensorType) < 0) return NULL;

    Py_INCREF(&PyTensorType);
    if (PyModule_AddObject(module, "Tensor", (PyObject*)&PyTensorType) < 0) {
        Py_DECREF(&PyTensorType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}