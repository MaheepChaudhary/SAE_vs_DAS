import sys
import os
import wandb
import pickle as pkl
import argparse
from transformers import BertTokenizer
import numpy as np
import random

parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from datasets import load_dataset
import random
from nnsight import LanguageModel 
import torch 
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

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation. A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.")

import logging
from transformers import logging as transformers_logging

# Suppress logging messages from the `transformers` library
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
