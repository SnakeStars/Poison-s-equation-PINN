import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# Выбор ресурса для обучения (автоматический)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Класс нейронной сети

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.flatten = nn.Flatten()
        self.tanh_layers_stack = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #1
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #2
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #1
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, dat):
        return self.tanh_layers_stack(dat)
    
model = NeuralNetwork().to(device)
print(model)

torch.save(model.state_dict(), 'Poison-s-PINN-start-weights.pth') # сохранить веса модели
#model.load_state_dict(torch.load('Poison-s-PINN-start-weights.pth', weights_only=True)) # загрузить веса модели

#  Задание параметров модели:

Q = [[0, 2], [0, 2]]                    # Borders
step = 150                              # points in one dim
EPOH = 100                              # study iterations
mode = 0                                # 1 - training, 0 - working on saved data (only weights and loss history saved!)

# Data

lossArr = []

# Создание сетки:

dat = []
for i in torch.linspace(Q[1][0], Q[1][1], step):
    dat.append(torch.linspace(Q[0][0], Q[0][1], step))
x = torch.cat(dat).unsqueeze(1).to(device)
dat = []
for i in torch.linspace(Q[0][0], Q[0][1], step):
    data = []
    for j in torch.linspace(Q[1][0], Q[1][1], step):
        data.append(i)
    dat.append(torch.tensor(data))
y = torch.cat(dat).unsqueeze(1).to(device)
t = torch.cat([x,y],dim=-1)

x_in = x[(x[:, 0] != Q[0][0]) & (x[:, 0] != Q[0][1]) & (y[:, 0] != Q[1][0]) & (y[:, 0] != Q[1][1])]
x_in.requires_grad = True
y_in = y[(y[:, 0] != Q[1][0]) & (y[:, 0] != Q[1][1]) & (x[:, 0] != Q[0][0]) & (x[:, 0] != Q[0][1])]
y_in.requires_grad = True
t_in = torch.cat([x_in,y_in],dim=-1)
# Создание шкалы загрузки:

pbar = tqdm(range(EPOH), desc='Training Progress')

# Оптимизатор и функция подсчета ошибки

metric_data = nn.MSELoss()
writer = SummaryWriter()
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)

# Уравнение функции

def pde(out, fx, fy):

    dudx = (torch.autograd.grad(out, fx, torch.ones_like(fx), create_graph=True,
                            retain_graph=True)[0])
    d2udx2 = (torch.autograd.grad(dudx, fx, torch.ones_like(fx), create_graph=True,
                            retain_graph=True)[0])
    
    dudy = (torch.autograd.grad(out, fy, torch.ones_like(fy), create_graph=True,
                            retain_graph=True)[0])
    d2udy2 = (torch.autograd.grad(dudy, fy, torch.ones_like(fy), create_graph=True,
                            retain_graph=True)[0])
    return d2udx2 + d2udy2

# Уравнение ошибки

def pdeLoss(t):
    out = model(t_in).to(device)
    f = pde(out, x_in, y_in)

    t_bc = torch.cat([t[(t[:,1] == Q[1][0]) & (t[:,0] != Q[0][0]) & (t[:,0] != Q[0][1])], t[(t[:,1] == Q[1][1]) & (t[:,0] != Q[0][0]) & (t[:,0] != Q[0][1])],t[(t[:,0] == Q[0][0])], t[(t[:,0] == Q[0][1])]])
    
    f_bc = model(t_bc).to(device)
    g_true = torch.mul( torch.sin(torch.mul(torch.pi,t_bc[:, 0].clone())) , torch.sin(torch.mul(torch.pi,t_bc[:, 1].clone()))  ).unsqueeze(1)
    f_true = torch.mul(-2, torch.mul(torch.pi ** 2, torch.mul( torch.sin(torch.mul(torch.pi,t_in[:, 0].clone())) ,torch.sin(torch.mul(torch.pi,t_in[:, 1].clone()))  ))).unsqueeze(1)

    
    loss_bc = metric_data(f_bc, g_true)
    loss_pde = metric_data(f, f_true)
    loss = loss_pde + 3*loss_bc
    lossArr.append(np.log10(loss.item()))

    return loss

# Функция тренировки нейросети

def train():

    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = pdeLoss(t)
            loss.backward()
            return loss

        optimizer.step(closure)
        if step % 2 == 0:
            current_loss = closure().item()
            pbar.set_description("Step: %d | Loss: %.7f" %
                                 (step, current_loss))
            writer.add_scalar('Loss/train', current_loss, step)

def show(x, y, z, arr, xlab):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.scatter(x,y,z,c=z, cmap='viridis',s=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(xlab, arr)
    plt.show()



if __name__ == "__main__":
    if mode:
        train()
        np.savetxt("loss.csv",lossArr, delimiter=",")
        torch.save(model.state_dict(), 'Poison-s-PINN-finish-weights.pth')
        show(x.cpu().detach().numpy(),y.cpu().detach().numpy(),model(t).to(device).cpu().detach().numpy(),lossArr,torch.arange(0,len(lossArr),1).cpu().numpy())
    else:
        model.load_state_dict(torch.load('Poison-s-PINN-finish-weights.pth', weights_only=True))
        model.eval()
        lossArr = np.genfromtxt("loss.csv", delimiter=",")
        show(x.cpu().detach().numpy(),y.cpu().detach().numpy(),model(t).to(device).cpu().detach().numpy(),lossArr,torch.arange(0,len(lossArr),1).cpu().numpy())

