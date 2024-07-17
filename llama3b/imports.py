import os
from sae.sae import Sae
from nnsight import LanguageModel 
import torch as t
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer