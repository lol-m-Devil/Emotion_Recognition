import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm

x = torch.randn(4, 3)
print(x)
y = F.softmax(x, dim = 0)
print(y)
z = F.softmax(x, dim = -1)
print(z)

