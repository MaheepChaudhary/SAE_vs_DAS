from pprint import pprint
import json
import argparse
from tqdm import tqdm
from pprint import pprint
import wandb
import random

from nnsight import LanguageModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer

import torch
import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import warnings
warnings.filterwarnings('ignore')


# Suppress specific warnings
warnings.filterwarnings("ignore", message="A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.")
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.")

import logging
from transformers import logging as transformers_logging

# Suppress logging messages from the `transformers` library
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
