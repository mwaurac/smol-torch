#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "ops.h"
#include "python_tensor.h"

static PyObject* PyTensor_add(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }

    if (!PyObject_IsInstance(a_obj, (PyObject*)&PyTensorType) ||
    !PyObject_IsInstance(b_obj, (PyObject*)&PyTensorType)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be Tensor objects");
        return NULL;
    }


    PyTensorObject* a = (PyTensorObject*)a_obj;
    PyTensorObject* b = (PyTensorObject*)b_obj;

    Tensor* result = add_tensor(a->tensor, b->tensor);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to add tensor");
        return NULL;
    }

    PyTensorObject* out = PyObject_New(PyTensorObject, &PyTensorType);
    if (!out) {
        tensor_free(result);
        return NULL;
    }
    out->tensor = result;
    return (PyObject*)out;
}

static PyMethodDef smol_torch_methods[] = {
    {"add", (PyCFunction)PyTensor_add, METH_VARARGS, "Add two tensors"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef smol_torch_module = {
    PyModuleDef_HEAD_INIT,
    "smol_torch",
    "A small torch-like library",
    -1,
    smol_torch_methods,
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