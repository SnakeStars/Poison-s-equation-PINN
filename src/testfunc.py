import torch
# -----------------------------------
EPOH = 100
# -----------------------------------
equalLoss = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

print(torch.mul(torch.sin(torch.pi * all_the_X),torch.sin(torch.pi * all_the_Y)))

