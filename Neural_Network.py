import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

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
    def __init__(self, hidden_size=8):
        super().__init__()
        self.flatten = nn.Flatten()
        self.tanh_layers_stack = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #1
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #2
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #3
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), #4
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, dat):
        return self.tanh_layers_stack(self.dat)
    
model = NeuralNetwork().to(device)
print(model)

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
    out = model(t).to(device)
    f1 = pde(out, x, y)
    g_true = torch.matmul(torch.mul(x, torch.pi).sin(), torch.mul(y, torch.pi).sin().unsqueeze(1)) # не доделал тензор с примененной функцией g

    loss_bc = metric_data(x0, x0_true) + metric_data(dx0dt, dx0dt_true.to(device))
    loss_pde = metric_data(f1, torch.zeros_like(f1))

    loss = loss_bc + loss_pde

    return loss


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
            pbar.set_description("Step: %d | Loss: %.6f" %
                                 (step, current_loss))
            writer.add_scalar('Loss/train', current_loss, step)




if __name__ == "__main__":

    #  Задание параметров модели:

    Q = [[0, 2], [0, 2]]                    # Borders
    step = 10                               # points in one dim
    EPOH = 1000                             # study iterations

    def f(x, y):
        return (-2) * (torch.pi ** 2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    

    def g(x,y):
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    
    def F(x,y, Q, res):
        return g(x,y) + (x - Q[0][0])*(Q[0][1] - x)*(y - Q[1][0])*(Q[1][1] - y) * res


    # Создание сетки:

    x = torch.linspace(Q[0][0], Q[0][1], step)
    x.requires_grad = True
    y = torch.linspace(Q[1][0], Q[1][1], step)
    y.requires_grad = True
    t = torch.cartesian_prod(x, y)


    # Создание шкалы загрузки:

    pbar = tqdm(range(EPOH), desc='Training Progress')


    # Оптимизатор и функция подсчета ошибки

    metric_data = nn.MSELoss()
    writer = SummaryWriter()
    optimizer = torch.optim.adam(model.parameters(), lr=0.1)

