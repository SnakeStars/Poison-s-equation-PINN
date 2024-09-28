import os
import torch

import torch.share
torch.autograd.set_detect_anomaly(True)
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import numpy as np
import optuna


# -----------------------------------
EPOH = 100
# -----------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric_data = nn.MSELoss()



# x_min, x_max - минимальное и максимальное значение по оси X
x_min, x_max = 0, 2

# y_min, y_max - минимальное и максимальное значение по оси Y
y_min, y_max = 0, 2

# Nx, Ny - количество точек по осям X и Y соответственно
Nx, Ny = 100, 100

# Создаем равномерно распределенные точки по оси X
x = torch.linspace(x_min, x_max, Nx).to(device)

# Создаем равномерно распределенные точки по оси Y
y = torch.linspace(y_min, y_max, Ny).to(device)

# Создаем сетку координат X и Y
# X и Y - это 2D тензоры, где каждый элемент представляет соответствующую координату
X, Y = torch.meshgrid(x, y)

# Создаем маску для выделения внутренних точек
# Изначально все точки помечены как False (граничные)
mask = torch.zeros_like(X, dtype=bool)

# Помечаем внутренние точки как True
# [1:-1, 1:-1] выбирает все элементы, кроме первого и последнего по обеим осям
mask[1:-1, 1:-1] = True

# Создаем тензор с внутренними точками
# X[mask] выбирает все точки, где mask == True
# flatten() преобразует результат в 1D тензор
# stack() объединяет координаты X и Y в один 2D тензор
interior_X = X[mask].flatten()
interior_Y = Y[mask].flatten()

interior_X.requires_grad = True
interior_Y.requires_grad = True

interior_points = torch.stack([interior_X, interior_Y], dim=1)

# Присвоение requres_grad всем координатам точек тензора

# Аналогично создаем тензор с граничными точками
# ~mask инвертирует маску, выбирая граничные точки
boundary_X = X[~mask].flatten()
boundary_Y = Y[~mask].flatten()

boundary_X.requires_grad = True
boundary_Y.requires_grad = True

boundary_points = torch.stack([boundary_X, boundary_Y], dim=1)

all_the_X = X.flatten()
all_the_Y = Y.flatten()

all_points = torch.stack([all_the_X, all_the_Y], dim=1)

class simpleModel(nn.Module):
  def __init__(self,
               hidden_size=16):
    super().__init__()
    self.layers_stack = nn.Sequential(
        nn.Linear(2, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #1
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), #2
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
    )

  def forward(self, x):
    return self.layers_stack(x)

def pde(out, t):

    dudt = torch.autograd.grad(out, [interior_X, interior_Y], grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True, allow_unused=True)

    dudx = dudt[0]

    dudy = dudt[1]
    print(dudt)
    x_result = torch.autograd.grad(dudx, interior_X, grad_outputs=torch.ones_like(dudx),create_graph=True, retain_graph=True, allow_unused=True)[0]
    y_result = torch.autograd.grad(dudy, interior_Y, grad_outputs=torch.ones_like(dudy), create_graph=True, retain_graph=True, allow_unused=True)[0]

    return x_result + y_result

def pdeLoss(model, lambd):
   
   out_inside = model(interior_points).to(device)
   out_border = model(boundary_points).to(device)


   u_inside = pde(out_inside, interior_points)
   u_border = pde(out_border, boundary_points)


   g = torch.zeros_like(u_border)
   f_inside = torch.mul(-2, torch.mul(torch.pi ** 2, torch.mul( torch.sin(torch.mul(torch.pi,interior_points[:, 0])) ,torch.sin(torch.mul(torch.pi,interior_points[:, 1]))  ))).unsqueeze(1)


   loss_PDE = metric_data(u_inside, f_inside)
   loss_BC = metric_data(u_border, g)
   loss = loss_PDE + lambd * loss_BC
   return loss

def train(model, lambd):
        pbar = tqdm(range(EPOH),desc='Training Progress')
        optimizer = torch.optim.LBFGS(model.parameters())
        for stepd in pbar:
            def closure():
                optimizer.zero_grad()
                loss = pdeLoss(model, lambd)
                loss.backward()
                return loss

            optimizer.step(closure)
            if stepd % 2 == 0:
                current_loss = closure().item()
                pbar.set_description("Lambda: %.4f | Step: %d | Loss: %.7f" %
                                 (lambd, stepd, los))
        pbar.clear()
def show(z):
    plt.style.use('_mpl-gallery')
    Z = np.reshape(z, (len(X), len(X)))
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16, 9))
    ax1.plot_surface(X,Y,Z, cmap='hot')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')


if __name__ == "__main__":
    neural_model = simpleModel().to(device)
    train(neural_model, 1)
    show(neural_model(all_points).to(device).cpu().detach().numpy())
    plt.show()