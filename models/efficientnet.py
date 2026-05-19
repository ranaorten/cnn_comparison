import torch
print(torch.__version__)
print("MPS var mı?:", torch.backends.mps.is_available())