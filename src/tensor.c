#include "../include/tensor.h"

#include <stdlib.h>
#include <string.h>

int get_tensor_dtype_size(dtype_t dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_FLOAT64: return sizeof(double);
        case DTYPE_INT32: return sizeof(int32_t);
        case DTYPE_INT64: return sizeof(int64_t);
        default: return 0;
    }
}
int64_t get_tensor_size(const int64_t* shape, const int ndim) {
    int64_t size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

void get_tensor_strides(const int64_t* shape, int64_t* strides, const int ndim) {
    strides[ndim - 1] = 1;

    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

Tensor* create_tensor(int64_t* shape, int ndim, dtype_t dtype) {
    Tensor* tensor = malloc(sizeof(Tensor));

    if (!tensor) return NULL;

    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->requires_grad = false;

    tensor->shape = (int64_t*)malloc(sizeof(int64_t) * ndim);
    memcpy(tensor->shape, shape, sizeof(int64_t) * ndim);

    tensor->size = get_tensor_size(shape, ndim);

    tensor->strides = (int64_t*)malloc(sizeof(int64_t) * ndim);
    get_tensor_strides(shape, tensor->strides, ndim);

    // alocate data
    const int dtype_size = get_tensor_dtype_size(dtype);
    tensor->data = malloc(dtype_size * tensor->size);
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }
    memset(tensor->data, 0, tensor->size * dtype_size);

    return tensor;
}

Tensor* create_tensor_with_data(const void* data, int64_t* shape, int ndim, dtype_t dtype) {
    Tensor* tensor = create_tensor(shape, ndim, dtype);
    if (!tensor) return NULL;

    const int dtype_size = get_tensor_dtype_size(dtype);
    memcpy(tensor->data, data, dtype_size * tensor->size);

    return tensor;
}

void tensor_free(Tensor* tensor) {
    if (!tensor) return;

    if (tensor->data) {
        free(tensor->data);
    }
    if (tensor->shape) free(tensor->shape);
    if (tensor->strides) free(tensor->strides);
    free(tensor);
}
