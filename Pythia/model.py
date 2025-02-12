from imports import *
from dataprocessing import *


class Probe(nn.Module):
    def __init__(self, activation_dim):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True)

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return logits


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
    def __init__(self, 
                DEVICE,
                dict_embed_path,
                attn_dict_path,
                mlp_dict_path,
                resid_dict_path,
                resid_layers,
                method,
                activation_dim,
                expansion_factor,
                epochs):
        
        super(my_model, self).__init__()
        
        # We have intergrated the sigmoid_mask from pyvene (https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/interventions.py) 
        
        
        # self.temperature = t.nn.Parameter(t.tensor(0.01))
        dict_id = 10
        dictionary_size = expansion_factor * activation_dim
        layer = 4
        self.resid_arr = [3,6,9,12,15]
        
        if method == "sae masking":
            sae_dim = (1,1,512*expansion_factor)
            self.l4_mask = t.nn.Parameter(t.zeros(sae_dim), requires_grad=True)
        elif method == "neuron masking":
            neuron_dim = (1,1,512)
            self.l4_mask = t.nn.Parameter(t.zeros(neuron_dim), requires_grad=True)
        elif method == "das masking":
            das_dim = (1,1,512)
            self.l4_mask = t.nn.Parameter(t.zeros(das_dim), requires_grad=True)
        elif method == "das sae masking":
            das_dim = (1,1,dictionary_size)
            self.l4_mask = t.nn.Parameter(t.zeros(das_dim), requires_grad=True)
        
        self.resid_layer = resid_layers
        self.method = method
        
        
        self.submodules = []
        self.dictionaries = {}
        self.model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map=DEVICE, dispatch=True)

        self.submodules.append(self.model.gpt_neox.embed_in)
        self.dictionaries[self.model.gpt_neox.embed_in] = AutoEncoder.from_pretrained(
            dict_embed_path + f'/{dict_id}_{dictionary_size}/ae.pt',
            device=DEVICE
        )
        for i in range(layer + 1):
            self.submodules.append(self.model.gpt_neox.layers[i].attention)
            self.dictionaries[self.model.gpt_neox.layers[i].attention] = AutoEncoder.from_pretrained(
                attn_dict_path + f'{i}/{dict_id}_{dictionary_size}/ae.pt',
                device=DEVICE
            )

            self.submodules.append(self.model.gpt_neox.layers[i].mlp)
            self.dictionaries[self.model.gpt_neox.layers[i].mlp] = AutoEncoder.from_pretrained(
                mlp_dict_path + f'{i}/{dict_id}_{dictionary_size}/ae.pt',
                device=DEVICE
            )

            self.submodules.append(self.model.gpt_neox.layers[i])
            self.dictionaries[self.model.gpt_neox.layers[i]] = AutoEncoder.from_pretrained(
                resid_dict_path + f'{i}/{dict_id}_{dictionary_size}/ae.pt',
                device=DEVICE
            )

        # self.probe = probe.requires_grad_(False)
        self.probe = Probe(512)
        self.probe.load_state_dict(t.load('cpu_probe.pth'))
        self.probe = self.probe.to(DEVICE)
        
        for param in self.probe.parameters():
            param.requires_grad = False
        
        self.das_layers = []
        
        # for layer in self.resid_layers:
        #     rotate_layer = RotateLayer(512)
        #     self.rotate_layer = t.nn.utils.parametrizations.orthogonal(rotate_layer)
        #     self.das_layers.append(self.rotate_layer)
        
        rotate_layer = RotateLayer(512)
        self.rotate_layer = t.nn.utils.parametrizations.orthogonal(rotate_layer)
        
        sae_rotate_layer = RotateLayer(expansion_factor * activation_dim)
        self.sae_rotate_layer = t.nn.utils.parametrizations.orthogonal(sae_rotate_layer)
        
        self.module_not_tuple = []
        
        # dummy_text = """The quick brown fox jumps over the lazy dog"""
        
        # with self.model.trace(dummy_text):
        #     for module in self.submodules:
        #         if type(module.output.shape) != tuple: # if tuple is true then we are reffering to the residual layer.
        #             self.module_not_tuple.append(module)
        
        # self.probe = Probe
        

    def forward(self,text, temperature):
        
        l4_mask_sigmoid = t.sigmoid(self.l4_mask / temperature)
        
        with self.model.trace(text) as tracer:
            
            assert self.resid_layer == 4
            assert self.resid_arr[self.resid_layer] == 15
                
            if self.method == "sae masking":
            
                # for layer in self.resid_layers:
                # dictionary = self.dictionaries[self.submodules[self.resid_arr[layer]]]    
                dictionary = self.dictionaries[self.submodules[self.resid_arr[self.resid_layer]]]
                # acts = self.submodules[self.resid_arr[layer]].output[0][:].clone().save()
                acts = self.submodules[self.resid_arr[self.resid_layer]].output[0][:].clone().save()
                cond = type(self.submodules[self.resid_arr[self.resid_layer]].output.shape) == tuple
                acts = dictionary.encode(acts).save()
                acts = l4_mask_sigmoid * acts
                acts = dictionary.decode(acts)
                # self.submodules[self.resid_arr[layer]].output[0][:] = acts.clone()
                self.submodules[self.resid_arr[self.resid_layer]].output[0][:] = acts.clone()
                final_acts = self.submodules[-1].output[0][:].save()
                
                
            elif self.method == "neuron masking":
                
                # for layer in self.resid_layers:
                
                dictionary = self.dictionaries[self.submodules[self.resid_arr[self.resid_layer]]]
                acts = self.submodules[self.resid_arr[self.resid_layer]].output[0][:].clone().save()
                cond = type(self.submodules[self.resid_arr[self.resid_layer]].output.shape) == tuple
                acts = l4_mask_sigmoid * acts
                self.submodules[self.resid_arr[self.resid_layer]].output[0][:] = acts.clone()
                final_acts = self.submodules[-1].output[0][:].save()
                
            elif self.method == "das masking":
                
                k = 0
                
                # for layer in self.resid_layers:
                    
                    # dictionary = self.dictionaries[self.submodules[self.resid_arr[layer]]]
                dictionary = self.dictionaries[self.submodules[self.resid_arr[self.resid_layer]]]
                # acts = self.submodules[self.resid_arr[layer]].output[0][:].clone().save()
                acts = self.submodules[self.resid_arr[self.resid_layer]].output[0][:].clone().save()
                cond = type(self.submodules[self.resid_arr[self.resid_layer]].output.shape) == tuple
                # self.rotate_layer = self.das_layers[k]
                acts = self.rotate_layer(acts)
                # acts = self.rotate_layer_1(acts)
                acts = l4_mask_sigmoid * acts
                acts = t.matmul(acts, self.rotate_layer.weight.T)
                # self.submodules[self.resid_arr[layer]].output[0][:] = acts.clone()
                self.submodules[self.resid_arr[self.resid_layer]].output[0][:] = acts.clone()
                final_acts = self.submodules[-1].output[0][:].save()
                k+=1
            
            elif self.method == "das sae masking":
                
                k = 0
                
                # for layer in self.resid_layers:
                    
                    # dictionary = self.dictionaries[self.submodules[self.resid_arr[layer]]]
                dictionary = self.dictionaries[self.submodules[self.resid_arr[self.resid_layer]]]
                # acts = self.submodules[self.resid_arr[layer]].output[0][:].clone().save()
                acts = self.submodules[self.resid_arr[self.resid_layer]].output[0][:].clone().save()
                cond = type(self.submodules[self.resid_arr[self.resid_layer]].output.shape) == tuple
                acts = dictionary.encode(acts).save()
                acts = self.sae_rotate_layer(acts)
                acts = l4_mask_sigmoid * acts
                acts = dictionary.decode(acts)
                acts_ = t.matmul(acts, self.rotate_layer.weight.T)
                # self.submodules[self.resid_arr[layer]].output[0][:] = acts.clone()
                self.submodules[self.resid_arr[self.resid_layer]].output[0][:] = acts_.clone()
                final_acts = self.submodules[-1].output[0][:].save()
                k+=1
            
            elif self.method == "sae masking":

                k = 0
                dictionary = self.dictionaries[self.submodules[self.resid_arr[self.resid_layer]]]
                acts = self.submodules[self.resid_arr[self.resid_layer]].output[0][:].clone().save()
                cond = type(self.submodules[self.resid_arr[self.resid_layer]].output.shape) == tuple
                acts = dictionary.encode(acts).save()
                # acts = self.sae_rotate_layer(acts)
                acts = l4_mask_sigmoid * acts
                acts = dictionary.decode(acts)
                acts_ = t.matmul(acts, self.rotate_layer.weight.T)
                self.submodules[self.resid_arr[self.resid_layer]].output[0][:] = acts_.clone()
                final_acts = self.submodules[-1].output[0][:].save()
                k+=1
        
        assert cond == True
        
        new_acts = final_acts.sum(1)
        acts = self.probe.net(new_acts).squeeze(-1)
            
        return acts, l4_mask_sigmoid
    