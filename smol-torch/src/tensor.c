#include "tensor.h"
#include <stdlib.h>
#include <string.h>

int get_tensor_dtype_size(Dtype dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_FLOAT64: return sizeof(double);
        case DTYPE_INT32: return sizeof(int32_t);
        case DTYPE_INT64: return sizeof(int64_t);
        default: return 0;
    }
}

int64_t get_tensor_size(const int64_t* shape, const int32_t ndim) {
    if (ndim <= 0) return 0;
    int64_t size = 1;
    for (int32_t i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return 0;
        size *= shape[i];
    }
    return size;
}

void get_tensor_strides(const int64_t* shape, int64_t* strides, const int32_t ndim) {
    if (ndim <= 0) return;
    strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

Tensor* create_tensor(int64_t* shape, int32_t ndim, Dtype dtype) {
    if (ndim <= 0 || get_tensor_dtype_size(dtype) == 0) return NULL;

    Tensor* tensor = malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device = BACKEND_CPU;
    tensor->requires_grad = false;
    tensor->offset = 0;

    tensor->shape = malloc(sizeof(int64_t) * ndim);
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, sizeof(int64_t) * ndim);

    tensor->size = get_tensor_size(shape, ndim);
    if (tensor->size == 0) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    tensor->strides = malloc(sizeof(int64_t) * ndim);
    if (!tensor->strides) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    get_tensor_strides(shape, tensor->strides, ndim);

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

Tensor* create_tensor_with_data(const void* data, int64_t* shape, int32_t ndim, Dtype dtype) {
    if (!data || ndim <= 0 || get_tensor_dtype_size(dtype) == 0) return NULL;

    Tensor* tensor = create_tensor(shape, ndim, dtype);
    if (!tensor) return NULL;

    const int dtype_size = get_tensor_dtype_size(dtype);
    memcpy(tensor->data, data, dtype_size * tensor->size);

    return tensor;
}

void tensor_free(Tensor* tensor) {
    if (!tensor) return;
    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}