import os
import sys

# Change the current working directory to '/content/sae'
os.chdir("/content/sae")
print(os.getcwd())

# Ensure the path is in sys.path
sys.path.append('/content/sae')

from sae.sae import Sae

saes = Sae.load_many_from_hub("EleutherAI/sae-llama-3-8b-32x")
saes["layers.10"]

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

with torch.inference_mode():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    outputs = model(**inputs, output_hidden_states=True)

    latent_acts = []
    for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
        latent_acts.append(sae.encode(hidden_state))

# Do stuff with the latent activations