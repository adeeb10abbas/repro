import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from scipy.io import loadmat
from math import floor

data = torch.Tensor(loadmat('3droad.mat')['data'])
