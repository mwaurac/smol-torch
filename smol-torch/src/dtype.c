#include "dtype.h"

#include <stdint.h>

int get_tensor_dtype_size(const Dtype dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_FLOAT64: return sizeof(double);
        case DTYPE_INT32: return sizeof(int32_t);
        case DTYPE_INT64: return sizeof(int64_t);
        default: return 0;
    }
}

static const int dtype_rank[DTYPE_COUNT] = {
    [DTYPE_INT32]   = 0,
    [DTYPE_INT64]   = 1,
    [DTYPE_FLOAT32] = 2,
    [DTYPE_FLOAT64] = 3,
};

const char* dtype_name(const Dtype dtype) {
    switch (dtype) {
        case DTYPE_INT32:   return "int32";
        case DTYPE_INT64:   return "int64";
        case DTYPE_FLOAT32: return "float32";
        case DTYPE_FLOAT64: return "float64";
        default:            return "unknown";
    }
}

Dtype promote(const Dtype a, const Dtype b) {
    const int rank_a = dtype_rank[a];
    const int rank_b = dtype_rank[b];
    const int rank = rank_a > rank_b ? rank_a : rank_b;

    for (int i = 0; i < DTYPE_COUNT; ++i) {
        if (dtype_rank[i] == rank) return (Dtype)i;
    }

    // fallback
    return DTYPE_FLOAT64;
}
