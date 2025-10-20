import smol_torch
# print(smol_torch.__version__)

tensor = smol_torch.Tensor( shape=[200, 2], dtype="float64" )
print(tensor)

a = smol_torch.Tensor([1, 3], shape=[1, 2], dtype="int32")
b = smol_torch.Tensor([2, 3], shape=[1, 2])
c = smol_torch.add(a, b)
print(c)