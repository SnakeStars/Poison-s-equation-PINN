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

step = 4
x = torch.linspace(0, 2, step)
x.requires_grad = True
y = torch.linspace(0, 2, step)
y.requires_grad = True
t = torch.cartesian_prod(x, y)

t_bc = torch.cat([t[(t[:,1] == Q[1][0]) & (t[:,0] != Q[0][0]) & (t[:,0] != Q[0][1])], 
                     t[(t[:,1] == Q[1][1]) & (t[:,0] != Q[0][0]) & (t[:,0] != Q[0][1])],
                     t[(t[:,0] == Q[0][0])], t[(t[:,0] == Q[0][1])]])
g_true = torch.mul( torch.sin(torch.mul(torch.pi,t_bc[:, 0].clone())) , torch.sin(torch.mul(torch.pi,t_bc[:, 1].clone()))  )
print(t_bc)
print(g_true)



# w = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# out = torch.ones_like(x)
# f = w*x+b
# dudx = (torch.autograd.grad(out, x, torch.ones_like(x), create_graph=True,
#                             retain_graph=True)[0])
# print(dudx)