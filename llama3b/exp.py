import os
import json
from sae.sae import Sae
from nnsight import LanguageModel 
import torch as t
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer

n_llama_model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map = t.device("cuda:1"))
sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.1").to(t.device("cuda:1"))

with n_llama_model.trace("my_name_is_maheep") as tracer:
    outputl1 = n_llama_model.model.layers[0].output[0].save()
   
sae_input = outputl1.squeeze(0)[-1].unsqueeze(0)
top_acts, top_indices  = sae.encode(sae_input)
d1 = sae.decode(top_acts, top_indices)

# Function to get the length of sentences
def get_sentence_length_counts(data):
    length_counts = {}
    for sublist in data:
        for sentence, label in sublist:
            length = len(n_llama_model.tokenizer(sentence)["input_ids"])
            if length in length_counts:
                length_counts[length] += 1
            else:
                length_counts[length] = 1
    return length_counts
# Get lengths of sentences

with open("final_data_continent.json","r") as f:
    continent_data = json.load(f)
    print(f"The total number of samples in continent are {len(continent_data)}")

with open("final_data_country.json","r") as f:
    country_data = json.load(f)
    print(f"The total number of samples in country are {len(country_data)}")

country_lengths = get_sentence_length_counts(country_data)

continent_lengths = get_sentence_length_counts(continent_data)

print(continent_lengths)
print(country_lengths)
