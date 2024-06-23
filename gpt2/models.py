from imports import *
from ravel_data_prep import *
from eval_gpt2 import *


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
            sae_dim = (1,1,512*self.expansion_factor)
            self.l4_mask = t.nn.Parameter(t.zeros(sae_dim), requires_grad=True)
        elif method == "neuron masking":
            # neuron_dim = (1,self.token_length_allowed, 768)
            neuron_dim = (1,1,768)
            self.l4_mask = t.nn.Parameter(t.zeros(neuron_dim), requires_grad=True)
        elif method == "das masking":
            das_dim = (1,1,512)
            self.l4_mask = t.nn.Parameter(t.zeros(das_dim), requires_grad=True)
        elif method == "vanilla":
            proxy_dim = (1,1,1)
            self.proxy = t.nn.Parameter(t.zeros(proxy_dim), requires_grad=True)
        # elif method == "das sae masking":
        #     das_dim = (1,1,dictionary_size)
        #     self.l4_mask = t.nn.Parameter(t.zeros(das_dim), requires_grad=True)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        
    def forward(self, source_ids, base_ids, temperature):
        
        l4_mask_sigmoid = t.sigmoid(self.l4_mask / temperature)
        
        if self.method == "neuron masking" or self.method == "sae masking" or self.method == "das masking":
            with self.model.trace() as tracer:
                
                with tracer.invoke(source_ids) as runner:

                    vector_source = self.model.transformer.h[self.layer_intervened].output

                with tracer.invoke(base_ids) as runner_:
                    
                    intermediate_output = self.model.transformer.h[self.layer_intervened].output[0].clone()
                    intermediate_output = (1 - self.l4_mask) * intermediate_output[:,self.intervened_token_idx,:].unsqueeze(0) + self.l4_mask * vector_source[0][:,self.intervened_token_idx,:].unsqueeze(0)
                    assert intermediate_output.shape == vector_source[0][:,self.intervened_token_idx,:].unsqueeze(0).shape == torch.Size([1, 1, 768])
                    # Create a new tuple with the modified intermediate_output
                    # modified_output = (intermediate_output,) + self.model.transformer.h[self.layer_intervened].output[1:]
                    self.model.transformer.h[self.layer_intervened].output[0][:,self.intervened_token_idx,:] = intermediate_output
                    
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
