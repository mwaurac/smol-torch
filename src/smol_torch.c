#include <Python.h>
#include <stddef.h>
#include <stdlib.h>

#include "../include/tensor.h"

typedef struct {
    PyObject_HEAD
    Tensor* tensor;
} PyTensorObject;

static PyTypeObject PyTensorType;

static void PyTensor_dealloc(PyTensorObject* self) {
    if (self->tensor) tensor_free(self->tensor);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyTensor_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyTensorObject* self = (PyTensorObject*)type->tp_alloc(type, 0);
    if(self) {
        self->tensor = NULL;
    }
    return (PyObject*)self;
}

static int PyTensor_init(PyTensorObject* self, PyObject* args, PyObject* kwds) {
    PyObject* shape_list;
    char* keywords[] = {"shape", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", keywords, &shape_list)) {
        return -1;
    }

    if(!PyList_Check(shape_list)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a list");
        return -1;
    }

    Py_ssize_t ndim = PyList_Size(shape_list);
    int64_t* shape = malloc(ndim * sizeof(int64_t));

    for(Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PyList_GetItem(shape_list, i);
        if(!PyLong_Check(item)) {
            free(shape);
            PyErr_SetString(PyExc_TypeError, "shape elements must be integers");
            return -1;
        }
        shape[i] = PyLong_AsLong(item);
    }

    self->tensor = create_tensor(shape, (int)ndim, DTYPE_FLOAT32);
    free(shape);

    if(!self->tensor) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create tensor");
        return -1;
    }

    return 0;
}

static PyObject* PyTensor_shape(PyTensorObject* self, PyObject* Py_UNUSED(ignored)) {
    if(!self->tensor) {
        Py_RETURN_NONE;
    }

    PyObject* shape_tuple = PyTuple_New(self->tensor->ndim);
    for(int i = 0; i < self->tensor->ndim; i++) {
        PyTuple_SetItem(shape_tuple, i, PyLong_FromLong(self->tensor->shape[i]));
    }
    return shape_tuple;
}

static PyMethodDef PyTensor_methods[] = {
    {"shape", (PyCFunction)PyTensor_shape, METH_NOARGS, "Returns the shape of the tensor"},
    {NULL}
};

static PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "smol_torch.Tensor",
    .tp_doc = "smol_torch Tensor object",
    .tp_basicsize = sizeof(PyTensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyTensor_new,
    .tp_init = (initproc)PyTensor_init,
    .tp_dealloc = (destructor)PyTensor_dealloc,
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
    if (PyType_Ready(&PyTensorType) < 0) return NULL;

    Py_INCREF(&PyTensorType);
    PyModule_AddObject(module, "Tensor", (PyObject*)&PyTensorType);

    return module;
}
