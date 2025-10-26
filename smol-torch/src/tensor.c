#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdarg.h>
#include <math.h>

#define MAX_PRINT_ELEMENTS 64
#define TRUNCATE_EDGE 5

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

static int ensure_capacity(char** buf, size_t* cap, size_t need) {
    if (need <= *cap) return 1;
    size_t newcap = *cap ? *cap * 2 : 1024;
    while (newcap < need) newcap *= 2;
    char* p = realloc(*buf, newcap);
    if (!p) return 0;
    *buf = p;
    *cap = newcap;
    return 1;
}

static int append_str(char** buf, size_t* pos, size_t* cap, const char* s) {
    size_t n = strlen(s);
    if (!ensure_capacity(buf, cap, *pos + n + 1)) return 0;
    memcpy(*buf + *pos, s, n);
    *pos += n;
    (*buf)[*pos] = '\0';
    return 1;
}

static int append_fmt(char** buf, size_t* pos, size_t* cap, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int needed = vsnprintf(NULL, 0, fmt, ap2);
    va_end(ap2);
    if (needed < 0) { va_end(ap); return 0; }
    if (!ensure_capacity(buf, cap, *pos + (size_t)needed + 1)) { va_end(ap); return 0; }
    vsnprintf(*buf + *pos, *cap - *pos, fmt, ap);
    *pos += (size_t)needed;
    va_end(ap);
    return 1;
}

static void format_element(char** buf, size_t* pos, size_t* cap, const Tensor* t, int64_t flat_idx) {
    if (t->dtype == DTYPE_FLOAT32) {
        float v = ((float*)t->data)[flat_idx];
        if (fabsf(v) < 1e-4 || fabsf(v) > 1e4) {
            append_fmt(buf, pos, cap, "%.4e", v);
        } else {
            append_fmt(buf, pos, cap, "%.4f", v);
        }
    } else if (t->dtype == DTYPE_FLOAT64) {
        double v = ((double*)t->data)[flat_idx];
        if (fabs(v) < 1e-4 || fabs(v) > 1e4) {
            append_fmt(buf, pos, cap, "%.4e", v);
        } else {
            append_fmt(buf, pos, cap, "%.4f", v);
        }
    } else if (t->dtype == DTYPE_INT32) {
        int32_t v = ((int32_t*)t->data)[flat_idx];
        append_fmt(buf, pos, cap, "%" PRId32, v);
    } else {
        int64_t v = ((int64_t*)t->data)[flat_idx];
        append_fmt(buf, pos, cap, "%" PRId64, v);
    }
}

static int print_recursive(char** buf, size_t* pos, size_t* cap, const Tensor* t,
                           int dim, int32_t ndim, int64_t* indices, int indent, bool truncate) {
    int64_t dim_size = t->shape[dim];
    if (dim == ndim - 1) {
        append_str(buf, pos, cap, "[");
        if (truncate && dim_size > 2 * TRUNCATE_EDGE) {

            for (int64_t i = 0; i < TRUNCATE_EDGE; ++i) {
                indices[dim] = i;
                int64_t flat = t->offset;
                for (int d = 0; d < ndim; ++d) flat += indices[d] * t->strides[d];
                if (i) append_str(buf, pos, cap, ", ");
                format_element(buf, pos, cap, t, flat);
            }
            append_str(buf, pos, cap, ", ...");

            for (int64_t i = dim_size - TRUNCATE_EDGE; i < dim_size; ++i) {
                indices[dim] = i;
                int64_t flat = t->offset;
                for (int d = 0; d < ndim; ++d) flat += indices[d] * t->strides[d];
                append_str(buf, pos, cap, ", ");
                format_element(buf, pos, cap, t, flat);
            }
        } else {

            for (int64_t i = 0; i < dim_size; ++i) {
                indices[dim] = i;
                int64_t flat = t->offset;
                for (int d = 0; d < ndim; ++d) flat += indices[d] * t->strides[d];
                if (i) append_str(buf, pos, cap, ", ");
                format_element(buf, pos, cap, t, flat);
            }
        }
        append_str(buf, pos, cap, "]");
        return 1;
    }

    append_str(buf, pos, cap, "[");
    append_str(buf, pos, cap, "\n");
    if (truncate && dim_size > 2 * TRUNCATE_EDGE) {

        for (int64_t i = 0; i < TRUNCATE_EDGE; ++i) {
            indices[dim] = i;
            for (int k = 0; k < indent + 1; ++k) append_str(buf, pos, cap, " ");
            if (!print_recursive(buf, pos, cap, t, dim + 1, ndim, indices, indent + 1, truncate)) return 0;
            append_str(buf, pos, cap, ",\n");
        }

        for (int k = 0; k < indent + 1; ++k) append_str(buf, pos, cap, " ");
        append_str(buf, pos, cap, "...\n");

        for (int64_t i = dim_size - TRUNCATE_EDGE; i < dim_size; ++i) {
            indices[dim] = i;
            for (int k = 0; k < indent + 1; ++k) append_str(buf, pos, cap, " ");
            if (!print_recursive(buf, pos, cap, t, dim + 1, ndim, indices, indent + 1, truncate)) return 0;
            if (i != dim_size - 1) append_str(buf, pos, cap, ",\n");
            else append_str(buf, pos, cap, "\n");
        }
    } else {

        for (int64_t i = 0; i < dim_size; ++i) {
            indices[dim] = i;
            for (int k = 0; k < indent + 1; ++k) append_str(buf, pos, cap, " ");
            if (!print_recursive(buf, pos, cap, t, dim + 1, ndim, indices, indent + 1, truncate)) return 0;
            if (i != dim_size - 1) append_str(buf, pos, cap, ",\n");
            else append_str(buf, pos, cap, "\n");
        }
    }
    for (int k = 0; k < indent; ++k) append_str(buf, pos, cap, " ");
    append_str(buf, pos, cap, "]");
    return 1;
}

char* tensor_to_string(const Tensor* t) {
    if (!t) return strdup("Tensor(NULL)");

    const char* dtype_str = dtype_name(t->dtype);

    char* buf = NULL;
    size_t cap = 0;
    size_t pos = 0;

    if (!append_fmt(&buf, &pos, &cap, "Tensor(shape=(")) goto error;
    for (int32_t i = 0; i < t->ndim; ++i) {
        if (i && !append_fmt(&buf, &pos, &cap, ", ")) goto error;
        if (!append_fmt(&buf, &pos, &cap, "%" PRId64, t->shape[i])) goto error;
    }
    if (!append_fmt(&buf, &pos, &cap, "), dtype=%s, data=\n", dtype_str)) goto error;

    if (t->size == 0) {
        if (!append_str(&buf, &pos, &cap, "[]")) goto error;
        return buf;
    }

    int32_t ndim = t->ndim;
    int64_t* indices = malloc(sizeof(int64_t) * ndim);
    if (!indices) goto error;

    bool truncate = t->size > MAX_PRINT_ELEMENTS;
    if (!print_recursive(&buf, &pos, &cap, t, 0, ndim, indices, 0, truncate)) {
        free(indices);
        goto error;
    }

    free(indices);
    return buf;

error:
    free(buf);
    return NULL;
}