# Импортируем библиотеку PyTorch
import torch

# x_min, x_max - минимальное и максимальное значение по оси X
x_min, x_max = 0, 2

# y_min, y_max - минимальное и максимальное значение по оси Y
y_min, y_max = 0, 2

# Nx, Ny - количество точек по осям X и Y соответственно
Nx, Ny = 100, 100

# Создаем равномерно распределенные точки по оси X
x = torch.linspace(x_min, x_max, Nx)

# Создаем равномерно распределенные точки по оси Y
y = torch.linspace(y_min, y_max, Ny)

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
interior_points = torch.stack([X[mask].flatten(), Y[mask].flatten()], dim=1)

# Аналогично создаем тензор с граничными точками
# ~mask инвертирует маску, выбирая граничные точки
boundary_points = torch.stack([X[~mask].flatten(), Y[~mask].flatten()], dim=1)

f = torch.mul(-2, torch.mul(torch.pi ** 2, torch.mul( torch.sin(torch.mul(torch.pi,interior_points[:, 0])) ,torch.sin(torch.mul(torch.pi,interior_points[:, 1]))  ))).unsqueeze(1)

print(interior_points)




# ----------------------------------------------------------------------




interior_points.requires_grad = True

f = interior_points * 2 + 1

dudx = torch.autograd.grad(f, interior_points, torch.ones_like(interior_points), create_graph=True, retain_graph=True)[0]

mask = torch.zeros_like(dudx, dtype=bool)

mask[:, 0] = True

print(dudx)



