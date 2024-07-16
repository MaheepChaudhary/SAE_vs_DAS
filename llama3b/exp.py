import os
from sae import Sae
import torch 
from transformers import AutoModel, AutoTokenizer

my_device = torch.device("cuda:1")
sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10", device = my_device)



#model_name = "meta-llama/Meta-Llama-3-8B"
#tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
# model = AutoModel.from_pretrained(model_name, token=True)
