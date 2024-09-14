import torch
import numpy as np

Q = [[0, 2], [0, 2]]                    # Borders
step = 4                                # points in one dim
EPOH = 1000                             # study iterations

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# dat = []
# for i in torch.linspace(Q[1][0], Q[1][1], step):
#     dat.append(torch.linspace(Q[0][0], Q[0][1], step))
# x = torch.cat(dat).unsqueeze(1).to(device)
# x.requires_grad = True
# dat = []
# for i in torch.linspace(Q[0][0], Q[0][1], step):
#     data = []
#     for j in torch.linspace(Q[1][0], Q[1][1], step):
#         data.append(i)
#     dat.append(torch.tensor(data, dtype=float))
# y = torch.cat(dat).unsqueeze(1).to(device)
# y.requires_grad = True
# t = torch.cat([x,y],dim=-1)
# print(t)
# print(torch.autograd.grad(t, x, torch.ones_like(torch.empty(16,2).to(device)), create_graph=True, retain_graph=True)[0].squeeze())

dat = []
for i in torch.linspace(Q[1][0], Q[1][1], step):
    dat.append(torch.linspace(Q[0][0], Q[0][1], step))
x = torch.cat(dat).unsqueeze(1).to(device)
x.requires_grad = True
dat = []
for i in torch.linspace(Q[0][0], Q[0][1], step):
    data = []
    for j in torch.linspace(Q[1][0], Q[1][1], step):
        data.append(i)
    dat.append(torch.tensor(data))
y = torch.cat(dat).unsqueeze(1).to(device)
y.requires_grad = True
t = torch.cat([x,y],dim=-1)

t_bc = torch.cat([t[(t[:,1] == Q[1][0]) & (t[:,0] != Q[0][0]) & (t[:,0] != Q[0][1])], 
                     t[(t[:,1] == Q[1][1]) & (t[:,0] != Q[0][0]) & (t[:,0] != Q[0][1])],
                     t[(t[:,0] == Q[0][0])], t[(t[:,0] == Q[0][1])]])
g_true = torch.mul( torch.sin(torch.mul(torch.pi,t_bc[:, 0].clone())) , torch.sin(torch.mul(torch.pi,t_bc[:, 1].clone()))  )
print(t_bc[:, 0].clone())
print(torch.sin(torch.mul(torch.pi,t_bc[:, 0].clone())))

# step = 4
# x = torch.tensor([1.,2.,1.,2.]).unsqueeze(1).to(device)
# y = torch.tensor([1.,1.,2.,2.]).unsqueeze(1).to(device)
# x.requires_grad = True
# y.requires_grad = True
# t = torch.cat([x,y],dim=-1)

# f = t

# print(f)
# t_bc = torch.cat([t[(t[:,1] == Q[1][0]) & (t[:,0] != Q[0][0]) & (t[:,0] != Q[0][1])], 
#                     t[(t[:,1] == Q[1][1]) & (t[:,0] != Q[0][0]) & (t[:,0] != Q[0][1])],
#                     t[(t[:,0] == Q[0][0])], t[(t[:,0] == Q[0][1])]])
# g_true = torch.mul( torch.sin(torch.mul(torch.pi,t_bc[:, 0].clone())) , torch.sin(torch.mul(torch.pi,t_bc[:, 1].clone()))  )

# print(torch.autograd.grad(f, x, torch.ones_like(torch.empty(4,2).to(device)), create_graph=True, retain_graph=True)[0].squeeze())



# w = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# out = torch.ones_like(x)
# f = w*x+b
# dudx = (torch.autograd.grad(out, x, torch.ones_like(x), create_graph=True,
#                             retain_graph=True)[0])
# print(dudx)