import os
from sae.sae import Sae

sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10")
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer

# Retrieve the token from Colab secrets
hf_token = os.getenv('HF_TOKEN')

# Log in with your token
login(token=hf_token)

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
model = AutoModel.from_pretrained(model_name, token="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")


# inputs = tokenizer("Hello, world!", return_tensors="pt")

# with t.inference_mode():
#     outputs = model(**inputs, output_hidden_states=True)

#     latent_acts = []
#     for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
#         latent_acts.append(sae.encode(hidden_state))
