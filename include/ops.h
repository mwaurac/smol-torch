#ifndef SMOL_TORCH_OPS_H
#define SMOL_TORCH_OPS_H
#include "tensor.h"

void t_add(const Tensor* a, const Tensor* b, Tensor* out);
Tensor* add_tensor(const Tensor* a, const Tensor* b);
Tensor* sub_tensor(Tensor* a, Tensor* b);


#endif //SMOL_TORCH_OPS_H