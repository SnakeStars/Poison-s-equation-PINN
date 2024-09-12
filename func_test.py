import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


print(torch.cartesian_prod(torch.arange(0, 1, 0.2), torch.arange(0, 1, 0.2)))

# w = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# out = torch.ones_like(x)
# f = w*x+b
# dudx = (torch.autograd.grad(out, x, torch.ones_like(x), create_graph=True,
#                             retain_graph=True)[0])
# print(dudx)