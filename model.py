from imports import *


class Probe(nn.Module):
    def __init__(self, activation_dim):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True)

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return logits




class my_model(nn.Module):
    def __init__(self, 
                DEVICE,
                probe,
                dict_embed_path = "/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/embed",
                attn_dict_path = "/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/attn_out_layer",
                mlp_dict_path = "/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/mlp_out_layer",
                resid_dict_path = "/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/resid_out_layer",
                activation_dim = 512,
                expansion_factor=64):
        
        super(my_model, self).__init__()
        
        # We have intergrated the sigmoid_mask from pyvene (https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/interventions.py) 
        
        
        embed_dim = (1,1,32768)
        self.temperature = t.nn.Parameter(t.tensor(0.01))
        dict_id = 10
        dictionary_size = expansion_factor * activation_dim
        layer = 4
        
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

        self.l4_mask = t.nn.Parameter(t.zeros(embed_dim), requires_grad=True)
        self.probe = probe.requires_grad_(False)
        
        self.module_not_tuple = []
        
        dummy_text = """The quick brown fox jumps over the lazy dog"""
        
        with self.model.trace(dummy_text):
            for module in self.submodules:
                if type(module.output.shape) != tuple: # if tuple is true then we are reffering to the residual layer.
                    self.module_not_tuple.append(module)
        
        # self.probe = Probe
        
        

    def forward(self,text):
        
        l4_mask_sigmoid = t.sigmoid(self.l4_mask / self.temperature.clone().detach().requires_grad_(True))
        
        with self.model.trace(text):
            
            dictionary = self.dictionaries[self.module_not_tuple[4]]
            acts = self.module_not_tuple[4].output.save()
            acts = dictionary.encode(acts).save()
            acts = l4_mask_sigmoid * acts
            acts = dictionary.decode(acts)
            self.module_not_tuple[4].output = acts
            final_acts = self.submodules[-1].output[0][:].save()

            
        new_acts = final_acts.sum(1)
        acts = self.probe.net(new_acts).squeeze(-1)
            
        return acts