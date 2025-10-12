import smol_torch
# print(smol_torch.__version__)

tensor = smol_torch.Tensor(data=[1,2,3, 3, 4, 5], shape=[1,2,3], dtype="float32")
print(tensor)
print(tensor.shape())
