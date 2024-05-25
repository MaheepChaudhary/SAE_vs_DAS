import torch
import blobfile as bf
import transformer_lens
from sparse_model import Autoencoder
from tqdm import tqdm
import pickle as pkl

device = "mps"

# Load the autoencoder
layer_index = 0  # in range(12)
autoencoder_input = ["mlp_post_act", "resid_delta_mlp"][0]
filename = f"az://openaipublic/sparse-autoencoder/gpt2-small/{autoencoder_input}/autoencoders/{layer_index}.pt"

with bf.BlobFile(filename, mode="rb") as f:
    print("Inside the blobfile")
    print()
    state_dict = torch.load(f)
    with open("state_dict.pkl", "wb") as f:
        pkl.dump(state_dict, f)
    autoencoder = Autoencoder.from_state_dict(state_dict)



# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
prompt = "This is an example of a prompt that"
print(prompt)
print()
tokens = model.to_tokens(prompt)  # (1, n_tokens)
print(model.to_str_tokens(tokens))
with torch.no_grad():
    print("Inside grad")
    logits, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)
if autoencoder_input == "mlp_post_act":
    input_tensor = activation_cache[f"blocks.{layer_index}.mlp.hook_post"]  # (n_tokens, n_neurons)
elif autoencoder_input == "resid_delta_mlp":
    input_tensor = activation_cache[f"blocks.{layer_index}.hook_mlp_out"]  # (n_tokens, n_residual_channels)

# Encode neuron activations with the autoencoder
device = next(model.parameters()).device
autoencoder.to(device)
with torch.no_grad():
    print(f"The input tensor is {input_tensor}")
    print()
    latent_activations = autoencoder.encode(input_tensor)  # (n_tokens, n_latents)
    print(f"The latent activation is {latent_activations}")
    print()
    original_activations = autoencoder.decode(latent_activations)  # (n_tokens, n_neurons)
    print(f"The original activation is {original_activations}")
    print()