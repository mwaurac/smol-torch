#ifndef SMOL_TORCH_TENSOR_H
#define SMOL_TORCH_TENSOR_H
#include <stdbool.h>
#include <stdint.h>

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_INT64
} Dtype;

typedef enum {
    BACKEND_CPU,
    BACKEND_CUDA
} Device;

typedef struct {
    void* data;
    int64_t* shape;
    int64_t* strides;
    int64_t size;
    int64_t offset;
    int32_t ndim;
    Dtype dtype;
    Device device;
    bool requires_grad;
} Tensor;

Tensor* create_tensor(int64_t* shape, int ndim, Dtype dtype);
Tensor* create_tensor_with_data(const void* data, int64_t* shape, int ndim, Dtype dtype);
void tensor_free(Tensor* tensor);

int get_tensor_dtype_size(Dtype dtype);
int64_t get_tensor_size(const int64_t* shape, int32_t ndim);
void get_tensor_strides(const int64_t* shape, int64_t* strides, int32_t ndim);
char* tensor_to_string(const Tensor* t);
#endif //SMOL_TORCH_TENSOR_H