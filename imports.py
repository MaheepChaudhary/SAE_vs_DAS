import sys
import os
import wandb
import pickle as pkl
import argparse

parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from datasets import load_dataset
import random
from nnsight import LanguageModel 
import torch as t
from torch import nn
# from attribution import patching_effect
from dictionary_learning import AutoEncoder, ActivationBuffer
# from dictionary_learning.dictionary import IdentityDict
# from dictionary_learning.interp import examine_dimension
# from dictionary_learning.utils import hf_dataset_to_generator
from tqdm import tqdm
import gc

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = dict(scan=True, validate=True)
else:
    tracer_kwargs = dict(scan=False, validate=False)

# model hyperparameters