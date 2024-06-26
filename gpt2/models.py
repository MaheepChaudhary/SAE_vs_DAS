from imports import *
from ravel_data_prep import *
from eval_gpt2 import *

torch.autograd.set_detect_anomaly(True)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to(cfg["device"])
    
    # inserted mask here
    def forward(self, x, mask, token_intervened_idx, bs):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        # Create a new tensor by modifying only the token_intervened_idx dimensions
        new_acts = acts.clone()  # Clone the original tensor to avoid modifying it in-place

        # Ensure token_intervened_idx is a tensor
        if isinstance(token_intervened_idx, int):
            token_intervened_idx = torch.tensor([token_intervened_idx], dtype=torch.long)

        # Reshape mask to match the new_acts dimensions
        mask = mask.view(-1, 6144)

        if bs == "base":
            new_acts[:, token_intervened_idx, :] = new_acts[:, token_intervened_idx, :] * mask
        elif bs == "source":
            new_acts[:, token_intervened_idx, :] = (1 - mask) * new_acts[:, token_intervened_idx, :]

        acts = new_acts

        x_reconstruct = new_acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
    
    def get_version(self):
        version_list = [int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)]
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    def save(self):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
        with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
            json.dump(cfg, f)
        print("Saved as version", version)
    
    @classmethod
    def load(cls, version):
        cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
        return self

    @classmethod
    def load_from_hf(cls, version, device_override=None):
        """
        Loads the saved autoencoder from HuggingFace. 
        
        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version=="run1":
            version = 25
        elif version=="run2":
            version = 47
        
        cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
        if device_override is not None:
            cfg["device"] = device_override

        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
        return self


class RotateLayer(t.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = t.empty(n, n)
        if init_orth:
            t.nn.init.orthogonal_(weight)
        self.weight = t.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        return t.matmul(x.to(self.weight.dtype), self.weight)


class my_model(nn.Module):
    def __init__(self, model, DEVICE, method, expansion_factor, token_length_allowed, layer_intervened, intervened_token_idx) -> None:
        super(my_model, self).__init__()
        
        self.model = model
        self.layer_intervened = layer_intervened
        self.intervened_token_idx = intervened_token_idx
        self.expansion_factor = expansion_factor
        self.token_length_allowed = token_length_allowed
        self.method = method
        
        if method == "sae masking":
            sae_dim = (1,1,6144)
            self.l4_mask = t.nn.Parameter(t.zeros(sae_dim), requires_grad=True)
            dic = torch.load("gpt2-small-sparse-autoencoders/gpt2-small_6144_mlp_out_0.pt", map_location=torch.device(DEVICE))
            cfg = {
                "dict_size": 6144,
                "act_size": 768,
                "l1_coeff": 0.001,
                "enc_dtype": "fp32",
                "seed": 0,
                "device": DEVICE,
                "model_batch_size": 1,
            }
            self.encoder_mlp_out_0 = AutoEncoder(cfg)
            self.encoder_mlp_out_0.load_state_dict(dic)
            
            for params in self.encoder_mlp_out_0.parameters():
                params.requires_grad = False
                
                
                
        
        elif method == "neuron masking":
            # neuron_dim = (1,self.token_length_allowed, 768)
            neuron_dim = (1,1,768)
            self.l4_mask = t.nn.Parameter(t.zeros(neuron_dim), requires_grad=True)
            
        elif method == "das masking":
            das_dim = (1,1,768)
            self.l4_mask = t.nn.Parameter(t.zeros(das_dim), requires_grad=True)
            rotate_layer = RotateLayer(768)
            self.rotate_layer = t.nn.utils.parametrizations.orthogonal(rotate_layer)
        elif method == "vanilla":
            proxy_dim = (1,1,1)
            self.proxy = t.nn.Parameter(t.zeros(proxy_dim), requires_grad=True)
        # elif method == "das sae masking":
        #     das_dim = (1,1,dictionary_size)
        #     self.l4_mask = t.nn.Parameter(t.zeros(das_dim), requires_grad=True)
        
        # for param in self.model.parameters():
        #     param.requires_grad = False
        
        
    def forward(self, source_ids, base_ids, temperature):
        
        l4_mask_sigmoid = t.sigmoid(self.l4_mask / temperature)
        
        
        
        if self.method == "neuron masking":
            with self.model.trace() as tracer:
                
                with tracer.invoke(source_ids) as runner:

                    vector_source = self.model.transformer.h[self.layer_intervened].output

                with tracer.invoke(base_ids) as runner_:
                    
                    intermediate_output = self.model.transformer.h[self.layer_intervened].output[0].clone()
                    intermediate_output = (1 - l4_mask_sigmoid) * intermediate_output[:,self.intervened_token_idx,:].unsqueeze(0) + l4_mask_sigmoid * vector_source[0][:,self.intervened_token_idx,:].unsqueeze(0)
                    assert intermediate_output.shape == vector_source[0][:,self.intervened_token_idx,:].unsqueeze(0).shape == torch.Size([1, 1, 768])
                    # Create a new tuple with the modified intermediate_output
                    # modified_output = (intermediate_output,) + self.model.transformer.h[self.layer_intervened].output[1:]
                    self.model.transformer.h[self.layer_intervened].output[0][:,self.intervened_token_idx,:] = intermediate_output
                    
                    intervened_base_predicted = self.model.lm_head.output.argmax(dim=-1).save()
                    intervened_base_output = self.model.lm_head.output.save()
                
                
            predicted_text = self.model.tokenizer.decode(intervened_base_predicted[0][-1])

            return intervened_base_output, predicted_text


        elif self.method == "das masking":
            
            l4_mask_sigmoid = t.sigmoid(self.l4_mask / temperature)
        
            with self.model.trace() as tracer:
                
                with tracer.invoke(source_ids) as runner:

                    vector_source = self.model.transformer.h[self.layer_intervened].output.save()

                with tracer.invoke(base_ids) as runner_:
                    
                    intermediate_output = self.model.transformer.h[self.layer_intervened].output[0].clone()
                    
                    # das 
                    vector_source_rotated = self.rotate_layer(vector_source[0][:,self.intervened_token_idx,:])
                    intermediate_output_rotated = self.rotate_layer(intermediate_output[:,self.intervened_token_idx,:])
                    
                    masked_intermediate_output_rotated = (1 - l4_mask_sigmoid) * intermediate_output_rotated 
                    masked_vector_source_rotated = l4_mask_sigmoid * vector_source_rotated
                    
                    masked_intermediate_output_unrotated = torch.matmul(masked_intermediate_output_rotated,self.rotate_layer.weight.T)
                    masked_vector_source_unrotated = torch.matmul(masked_vector_source_rotated,self.rotate_layer.weight.T)
                    
                    iia_vector = masked_intermediate_output_unrotated + masked_vector_source_unrotated
                    
                    # intermediate_output = (1 - self.l4_mask) * intermediate_output[:,self.intervened_token_idx,:].unsqueeze(0) + self.l4_mask * vector_source[0][:,self.intervened_token_idx,:].unsqueeze(0)
                    assert iia_vector.shape == vector_source[0][:,self.intervened_token_idx,:].unsqueeze(0).shape == torch.Size([1, 1, 768])
                    # Create a new tuple with the modified intermediate_output
                    # modified_output = (intermediate_output,) + self.model.transformer.h[self.layer_intervened].output[1:]
                    self.model.transformer.h[self.layer_intervened].output[0][:,self.intervened_token_idx,:] = iia_vector
                    
                    intervened_base_predicted = self.model.lm_head.output.argmax(dim=-1).save()
                    intervened_base_output = self.model.lm_head.output.save()
                
                
            predicted_text = self.model.tokenizer.decode(intervened_base_predicted[0][-1])

            return intervened_base_output, predicted_text

        elif self.method == "sae masking":
            
            with self.model.trace() as tracer:
                
                with tracer.invoke(source_ids) as runner:

                    source = self.model.transformer.h[self.layer_intervened].output

                with tracer.invoke(base_ids) as runner_:
                    
                    base = self.model.transformer.h[self.layer_intervened].output[0].clone()
                    _, source_masked, _, _, _ = self.encoder_mlp_out_0(source[0], l4_mask_sigmoid, self.intervened_token_idx, "base")
                    _, base_masked, _, _, _ = self.encoder_mlp_out_0(base, l4_mask_sigmoid, self.intervened_token_idx, "source")
                    # intermediate_output = (1 - l4_mask_sigmoid) * intermediate_output[:,self.intervened_token_idx,:].unsqueeze(0) + l4_mask_sigmoid * vector_source[0][:,self.intervened_token_idx,:].unsqueeze(0)
                    
                    base_masked_ind = base_masked[:,self.intervened_token_idx,:]
                    source_masked_ind = source_masked[:,self.intervened_token_idx,:]
                    
                    iia_vector = base_masked_ind + source_masked_ind
                    # assert iia_vector.unsqueeze(0).shape == source[0][:,self.intervened_token_idx,:].unsqueeze(0).shape == torch.Size([1, 1, 768])
                    # Create a new tuple with the modified intermediate_output
                    # modified_output = (intermediate_output,) + self.model.transformer.h[self.layer_intervened].output[1:]
                    self.model.transformer.h[self.layer_intervened].output[0][:,self.intervened_token_idx,:] = iia_vector
                    # self.model.transformer.h[self.layer_intervened].output[0][:, self.intervened_token_idx,:] = source_masked[:,self.intervened_token_idx,:]
                    
                    intervened_base_predicted = self.model.lm_head.output.argmax(dim=-1).save()
                    intervened_base_output = self.model.lm_head.output.save()
                
                
            predicted_text = self.model.tokenizer.decode(intervened_base_predicted[0][-1])

            return intervened_base_output, predicted_text


        elif self.method == "vanilla":
            intervened_token_idx = -8
            with self.model.trace() as tracer:
                
                with tracer.invoke(source_ids) as runner:

                    vector_source = self.model.transformer.h[self.layer_intervened].output

                with tracer.invoke(base_ids) as runner_:
                    
                    self.model.transformer.h[self.layer_intervened].output[0][:, intervened_token_idx, :] = vector_source[0][:,intervened_token_idx,:]
                    
                    intervened_base_predicted = self.model.lm_head.output.argmax(dim=-1).save()
                    intervened_base_output = self.model.lm_head.output.save()
                
            predicted_text = self.model.tokenizer.decode(intervened_base_predicted[0][-1])
            

            return intervened_base_output, predicted_text
