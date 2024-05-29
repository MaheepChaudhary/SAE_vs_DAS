import torch
from imports import *
import blobfile as bf
import transformer_lens
from sparse_autoencoder import Autoencoder
from tqdm import tqdm
import pickle as pkl

DEVICE = device = "cpu"


# Load the autoencoder
layer_index = 11  # in range(12)
autoencoder_input = ["mlp_post_act", "resid_delta_mlp"][1]
filename = f"az://openaipublic/sparse-autoencoder/gpt2-small/{autoencoder_input}/autoencoders/{layer_index}.pt"


# So we can say that the last residual layer to be used is the 11th residual layer, which will be now used to train the binary mask for ravel.

with bf.BlobFile(filename, mode="rb") as f:
    print("Inside the blobfile")
    print()
    state_dict = torch.load(f)
    with open("state_dict.pkl", "wb") as f:
        pkl.dump(state_dict, f)
    autoencoder = Autoencoder.from_state_dict(state_dict)

g

# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
prompt = "This is an example of a prompt that"
tokens = model.to_tokens(prompt)  # (1, n_tokens)


# Making the model for ravel dataset. 

class gpt2_ravel_model(nn.Module):
    
    def __init__(self, model, autoencoder, expansion_factor=4):
        super().__init__()
        
        self.model = model
        self.autoencoder = autoencoder
        # if method == "sae masking":
        sae_dim = (1,1,512*expansion_factor)
        self.l4_mask = t.nn.Parameter(t.zeros(sae_dim), requires_grad=True)
        
        
        
    def forward(self, tokens, temperature):
        with torch.no_grad():
            logits, activation_cache = self.model.run_with_cache(tokens, remove_batch_dim=True)
        if autoencoder_input == "mlp_post_act":
            input_tensor = activation_cache[f"blocks.{layer_index}.mlp.hook_post"]
        elif autoencoder_input == "resid_delta_mlp":
            input_tensor = activation_cache[f"blocks.{layer_index}.hook_mlp_out"]
        latent_activations = self.autoencoder.encode(input_tensor)
        l4_mask_sigmoid = t.sigmoid(self.l4_mask / temperature)
        latent_activations = l4_mask_sigmoid * latent_activations
        input_tensor = self.autoencoder.decode(latent_activations)
        return input_tensor
    




total_step = 0
# target_total_step = len(train_batches) * epochs
temperature_start = 50.0
temperature_end = 0.1
# temperature_schedule = (
#     t.linspace(temperature_start, temperature_end, target_total_step)
#     .to(t.bfloat16)
#     .to(DEVICE)
# )


