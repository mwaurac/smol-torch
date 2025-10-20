#ifndef SMOL_TORCH_DTYPE_H
#define SMOL_TORCH_DTYPE_H

typedef enum {
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_COUNT
} Dtype;

int get_tensor_dtype_size(Dtype dtype);

Dtype promote(Dtype a, Dtype b);
const char* dtype_name(Dtype dtype);

#endif // SMOL_TORCH_DTYPE_H
