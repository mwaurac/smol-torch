#ifndef SMOL_TORCH_TENSOR_H
#define SMOL_TORCH_TENSOR_H
#include <stdbool.h>
#include <stdint.h>

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_INT64
} dtype_t;

typedef struct {
    void* data;
    int64_t* shape;
    int64_t* strides;
    int64_t size;
    int ndim;
    dtype_t dtype;
    bool requires_grad;
} Tensor;

Tensor* create_tensor(int64_t* shape, int ndim, dtype_t dtype);
Tensor* create_tensor_with_data(const void* data, int64_t* shape, int ndim, dtype_t dtype);
void tensor_free(Tensor* tensor);
#endif //SMOL_TORCH_TENSOR_H