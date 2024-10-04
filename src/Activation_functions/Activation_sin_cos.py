import torch
import torch.nn as nn
from torch import cos, sin
# Inherit from Function
    
class Sin(nn.Module):
    
    def forward(self, input):
       return torch.sin(input)
    
class Cos(nn.Module):
    
    def forward(self, input):
       return torch.cos(input)
