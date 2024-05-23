import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchtyping import TensorType

import torch as t
from jaxtyping import Float, Int
from torch import nn, Tensor
import einops
from dataclasses import dataclass

import transformer_lens.utils as utils
from collections import Counter

import plotly.graph_objects as go
import plotly.express as px
import csv
from tqdm import tqdm

import time

# seconds passed since epoch
seconds = time.time()

# convert the time in seconds since the epoch to a readable format
local_time = time.ctime(seconds)

local_time_arr = local_time.split()[2:-1]
first_element = local_time_arr[0]
second_element = "".join(local_time_arr[-1].split(":")[0:-1])


name = "_".join([first_element, second_element])

