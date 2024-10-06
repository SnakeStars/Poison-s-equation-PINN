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
from Activation_functions.Activation_sin_cos import Sin, Cos


# -----------------------------------
EPOH = 20000
research = 1                        # 0 if show result, 1 if start research
# -----------------------------------
equalLoss = []
Loss = []
current_loss = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric_data = nn.MSELoss()


global_Loss = 0



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
        Sin(),
        nn.Linear(hidden_size, hidden_size), #1
        Sin(),
        nn.Linear(hidden_size, hidden_size), #2
        Sin(),
        nn.Linear(hidden_size, 1),
    )

  def forward(self, x):
    return self.layers_stack(x)

def pde(out, t, tensor_X, tensor_Y):
    dudt = torch.autograd.grad(out, [tensor_X, tensor_Y], grad_outputs=torch.ones_like(out), create_graph=True)

    dudx = dudt[0]

    dudy = dudt[1]
    x_result = torch.autograd.grad(dudx, tensor_X, grad_outputs=torch.ones_like(dudx),create_graph=True, allow_unused=True)[0]
    y_result = torch.autograd.grad(dudy, tensor_Y, grad_outputs=torch.ones_like(dudy), create_graph=True, allow_unused=True)[0]

    return x_result + y_result

def equal_f():
    return torch.mul(torch.sin(torch.pi * all_the_X),torch.sin(torch.pi * all_the_Y)).to(device)

def pdeLoss(model, lambd):
   
   out_inside = model(interior_points).to(device)
   out_border = model(boundary_points).to(device)


   u_inside = pde(out_inside, interior_points, interior_X, interior_Y)


   g = torch.zeros_like(out_border)
   f_inside = torch.mul(-2, torch.mul(torch.pi ** 2, torch.mul( torch.sin(torch.mul(torch.pi,interior_points[:, 0])) ,torch.sin(torch.mul(torch.pi,interior_points[:, 1]))  )))

   loss_PDE = metric_data(u_inside, f_inside)
   loss_BC = metric_data(out_border, g)
   loss = loss_PDE + lambd * loss_BC
   return loss

def train(model, lambd, trial=None):
        pbar = tqdm(range(EPOH),desc='Training Progress')
        optimizer = torch.optim.Adam(model.parameters())
        check = 0
        for step in pbar:
            def closure():
                global current_loss
                optimizer.zero_grad()
                loss = pdeLoss(model, lambd)
                current_loss = loss.item()
                loss.backward()
                return loss
            
            equal_loss = torch.norm(equal_f() - model(all_points).to(device).squeeze(1), p=float('inf')).item()

            optimizer.step(closure)

            if trial != None:
                trial.report(equal_loss, step)

                if trial.should_prune():
                    pbar.clear()
                    raise optuna.TrialPruned()
            if step >= 18000 and check:
                optimizer = torch.optim.Adam(model.parameters(), 1e-4)
                check = 1
            if trial == None:
                equalLoss.append(equal_loss)
                Loss.append(current_loss)

            if step % 2 == 0:
                pbar.set_description("Step: %d | Loss: %.7f | Lambda: %.6f" %
                                 (step, current_loss, lambd))

        pbar.clear()
        if trial != None:
            return equal_loss

def show(z):
    plt.style.use('_mpl-gallery')
    Z = np.reshape(z, (len(X), len(X)))
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16, 9))
    ax1.plot_surface(X.cpu(),Y.cpu(),Z, cmap='hot')
    #ax1.plot_surface(X.cpu(),Y.cpu(), np.reshape(equal_f().cpu(), (len(X), len(X))))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    fig2, ax2 = plt.subplots()

    fs = 12
    margins = {
    "left"   : 0.040,
    "bottom" : 0.060,
    "right"  : 0.950,
    "top"    : 0.950   
    }
    fig2.subplots_adjust(**margins) 

    ax2=plt.gca()
    ax2.plot(equalLoss)
    ax2.set_yscale('log')

    fig3, ax3 = plt.subplots()

    fs = 12
    margins = {
    "left"   : 0.040,
    "bottom" : 0.060,
    "right"  : 0.950,
    "top"    : 0.950   
    }
    fig3.subplots_adjust(**margins) 

    ax3=plt.gca()
    ax3.plot(Loss)
    ax3.set_yscale('log')
    ax3.set_ylabel("LOSS")

def objective(trial):
    neural_model = simpleModel().to(device)
    neural_model.load_state_dict(torch.load('neural_model_weigths.pth', map_location=torch.device(device), weights_only=True))
    x = trial.suggest_float('x', 1e-6, 1e6, log=True)
    err = train(neural_model, x, trial)
    return err

def study_show(study):
    ax1 = optuna.visualization.matplotlib.plot_intermediate_values(study)
    ax2 = optuna.visualization.matplotlib.plot_intermediate_values(study)
    ax3 = optuna.visualization.matplotlib.plot_intermediate_values(study)
    ax4 = optuna.visualization.matplotlib.plot_optimization_history(study)
    ax5 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()

    ax1.set_yscale('log')
    ax1.set_xlim(-5,1000)
    ax1.set_title("Hyperparameter selection")
    ax1.set_xlabel("EPOH")
    ax1.set_ylabel("Analytical deviation")

    ax2.set_yscale('log')
    ax2.set_xlim(EPOH - 1000,EPOH+5)
    ax2.set_title("Hyperparameter selection")
    ax2.set_xlabel("EPOH")
    ax2.set_ylabel("Analytical deviation")

    ax3.set_yscale('log')
    ax3.set_title("Hyperparameter selection")
    ax3.set_xlabel("EPOH")
    ax3.set_ylabel("Analytical deviation")

    ax4.set_yscale('log')
    ax4.set_title("Hyperparameter selection")
    ax4.set_ylabel("Analytical deviation")

    ax5.set_yscale('log')
    ax5.set_title("Hyperparameter selection")

    ax1.get_figure().set_size_inches(16, 9)
    ax2.get_figure().set_size_inches(16, 9)
    ax3.get_figure().set_size_inches(16, 9)
    ax4.get_figure().set_size_inches(16, 9)
    ax5.get_figure().set_size_inches(16, 9)

    ax1.get_figure().savefig('Losses_begin.png')
    ax2.get_figure().savefig('Losses_end.png')
    ax3.get_figure().savefig('Losses_total.png')
    ax4.get_figure().savefig('history_losses.png')
    ax5.get_figure().savefig('paretto.png')

    f = open("Best value.txt", "w")
    f.write(str(study.best_params["x"]))
    f.close()

    plt.show()

if __name__ == "__main__":
    if research:
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10, n_warmup_steps=2500, interval_steps=300
                ),
                )
        study.optimize(objective, n_trials=200)
        study_show(study)
    else:
        neural_model = simpleModel().to(device)
        neural_model.load_state_dict(torch.load('neural_model_weigths.pth', map_location=torch.device(device), weights_only=True))
        train(neural_model, 753)
        show(neural_model(all_points).to(device).cpu().detach().numpy())
        plt.show()

    