import torch
X_tensor = torch.tensor(torch.randn(3,5,5), dtype=torch.float32)
y_tensor = torch.tensor(torch.randn(5,6), dtype=torch.float32)

c= torch.einsum("bst,td->sd",X_tensor,y_tensor)
print(c.size())
print()