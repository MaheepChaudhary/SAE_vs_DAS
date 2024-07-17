import os
from sae.sae import Sae
from nnsight import LanguageModel 
import torch as t
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer

n_llama_model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map = t.device("cuda:1"))
sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.1").to(t.device("cuda:1"))
print(n_llama_model)

with n_llama_model.trace("my_name_is_maheep") as tracer:
    outputl1 = n_llama_model.model.layers[0].output[0].save()
   
sae_input = outputl1.squeeze(0)[-1].unsqueeze(0)
print(sae_input.shape)
top_acts, top_indices  = sae.encode(sae_input)
d1 = sae.decode(top_acts, top_indices)

print(d1.shape)
#print(top_indices)
print(outputl1.shape)


model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
model = AutoModel.from_pretrained(model_name, token="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")

inputs = tokenizer("Hello, world!", return_tensors="pt")

with t.inference_mode():
    outputs = model(**inputs, output_hidden_states=True)

    latent_acts = []
#     for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
#1         latent_acts.append(sae.encode(hidden_state))
