import torch

Q = [[0, 2], [0, 2]]                    # Borders
step = 10                               # points in one dim
EPOH = 1000                             # study iterations

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

x = torch.linspace(Q[0][0], Q[0][1], step)
y = torch.linspace(Q[1][0], Q[1][1], step)
g = torch.matmul(torch.mul(x, torch.pi).sin(), torch.mul(y, torch.pi).sin())
data = [torch.mul(y, torch.pi).sin() for i in range(10)]
print(torch.mul(data, torch.pi).sin())
# w = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# out = torch.ones_like(x)
# f = w*x+b
# dudx = (torch.autograd.grad(out, x, torch.ones_like(x), create_graph=True,
#                             retain_graph=True)[0])
# print(dudx)