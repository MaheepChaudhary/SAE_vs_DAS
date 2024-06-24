from imports import *
from ravel_data_prep import *
from eval_gpt2 import *

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
            sae_dim = (1,1,512*self.expansion_factor)
            self.l4_mask = t.nn.Parameter(t.zeros(sae_dim), requires_grad=True)
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
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        
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
            pass

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
