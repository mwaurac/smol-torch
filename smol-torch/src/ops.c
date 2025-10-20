#include "ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

bool tensor_same_shape(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim) return false;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

#define DEFINE_ADD_OP(DTYPE_ENUM, C_TYPE)                                      \
  case DTYPE_ENUM: {                                                           \
    C_TYPE *a_data = (C_TYPE *)a->data;                                        \
    C_TYPE *b_data = (C_TYPE *)b->data;                                        \
    C_TYPE *out_data = (C_TYPE *)out->data;                                    \
    for (size_t i = 0; i < out->size; i++) {                                   \
      out_data[i] = (C_TYPE)a_data[i] + (C_TYPE)b_data[i];                     \
    }                                                                          \
    break;                                                                     \
  }

void t_add(const Tensor *a, const Tensor *b, Tensor *out) {
  switch (out->dtype) {
    DEFINE_ADD_OP(DTYPE_FLOAT32, float)
    DEFINE_ADD_OP(DTYPE_FLOAT64, double)
    DEFINE_ADD_OP(DTYPE_INT32, int32_t)
    DEFINE_ADD_OP(DTYPE_INT64, int64_t)
  default:
    fprintf(stderr, "Unsupported dtype for addition: %s\n", dtype_name(out->dtype));
    break;
  }
}

Tensor* add_tensor(const Tensor* a, const Tensor* b) {
    if (!tensor_same_shape(a, b)) {
        fprintf(stderr, "Incompatible shapes for tensor addition\n");
        return NULL;
    }

    if (a->device != b->device) {
        printf("Error: Tensors must be on same device\n");
        return NULL;
    }

    const Dtype o_dtype = promote(a->dtype, b->dtype);
    Tensor* out = create_tensor(a->shape, a->ndim, o_dtype);
    if (!out) return NULL;

    t_add(a, b, out);

    out->device = a->device;

    return out;
}
