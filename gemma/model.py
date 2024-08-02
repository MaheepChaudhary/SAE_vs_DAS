from huggingface_hub import hf_hub_download
from imports import *
from transformer_lens.hook_points import HookedRootModule, HookPoint


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
    def __init__(
        self,
        model,
        DEVICE,
        method,
        expansion_factor,
        token_length_allowed,
        layer_intervened,
        intervened_token_idx,
        batch_size,
    ) -> None:
        super(my_model, self).__init__()

        self.model = model
        self.layer_intervened = t.tensor(layer_intervened, dtype=t.int32, device=DEVICE)
        self.intervened_token_idx = t.tensor(
            intervened_token_idx, dtype=t.int32, device=DEVICE
        )
        self.intervened_token_idx = intervened_token_idx
        self.expansion_factor = expansion_factor
        self.token_length_allowed = token_length_allowed
        self.method = method
        self.batch_size = batch_size

        self.DEVICE = DEVICE
        print(model)

        # if method == "sae masking":
        #     sae_dim = (1, 24576)
        #     self.l4_mask = t.nn.Parameter(t.zeros(sae_dim), requires_grad=True)
        #
        #     self.sae_neel, cfg_dict, sparsity = SAE.from_pretrained(
        #         release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        #         sae_id=f"blocks.{self.layer_intervened}.hook_resid_pre",  # won't always be a hook point
        #     )
        #     for params in self.sae_neel.parameters():
        #         params.requires_grad = False
        #
        if method == "neuron masking":
            # neuron_dim = (1,self.token_length_allowed, 768)
            neuron_dim = (1, 4096)
            self.l4_mask = t.nn.Parameter(
                t.zeros(neuron_dim, device=DEVICE), requires_grad=True
            )
            self.l4_mask = self.l4_mask.to(DEVICE)

        elif method == "das masking":
            das_dim = (1, 4096)
            self.l4_mask = t.nn.Parameter(
                t.zeros(das_dim, device=DEVICE), requires_grad=True
            )
            rotate_layer = RotateLayer(4096)
            self.rotate_layer = t.nn.utils.parametrizations.orthogonal(rotate_layer)

    def forward(self, source_ids, base_ids, temperature):
        l4_mask_sigmoid = t.sigmoid(self.l4_mask / temperature)
        # l4_mask_sigmoid = self.l4_mask
        if self.method == "neuron masking":
            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:
                    vector_source = self.model.model.layers[
                        self.layer_intervened
                    ].output[0]

                with tracer.invoke(base_ids) as runner_:
                    intermediate_output = (
                        self.model.model.layers[self.layer_intervened].output[0].clone()
                    )
                    intermediate_output = (1 - l4_mask_sigmoid) * intermediate_output[
                        :, self.intervened_token_idx, :
                    ] + l4_mask_sigmoid * vector_source[:, self.intervened_token_idx, :]
                    assert (
                        intermediate_output.squeeze(1).shape
                        == vector_source[:, self.intervened_token_idx, :].shape
                        == torch.Size([self.batch_size, 4096])
                    )
                    self.model.model.layers[self.layer_intervened].output[0][
                        :, self.intervened_token_idx, :
                    ] = intermediate_output.squeeze(1)
                    # self.model.model.layers[self.layer_intervened].output[0][:,self.intervened_token_idx,:] = vector_source[:,self.intervened_token_idx,:]

                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = []
            for index in range(intervened_base_output.shape[0]):
                predicted_text.append(
                    self.model.tokenizer.decode(
                        intervened_base_output[index].argmax(dim=-1)
                    ).split()[-1]
                )

            return intervened_base_output, predicted_text

        elif self.method == "das masking":

            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:

                    vector_source = self.model.transformer.h[
                        self.layer_intervened
                    ].output.save()

                with tracer.invoke(base_ids) as runner_:

                    intermediate_output = (
                        self.model.transformer.h[self.layer_intervened].output[0].save()
                    )
                    # print("Intermediate shape",intermediate_output.shape)

                    # das
                    assert (
                        vector_source[0][:, self.intervened_token_idx, :].shape
                        == intermediate_output[:, self.intervened_token_idx, :].shape
                        == torch.Size([self.batch_size, 4096])
                    )
                    vector_source_rotated = self.rotate_layer(
                        vector_source[0][:, self.intervened_token_idx, :]
                    )
                    cloned_intermediate_output = intermediate_output.clone()
                    intermediate_output_rotated = self.rotate_layer(
                        cloned_intermediate_output[:, self.intervened_token_idx, :]
                    )
                    assert (
                        intermediate_output_rotated.shape
                        == vector_source_rotated.shape
                        == torch.Size([self.batch_size, 4096])
                    )
                    assert l4_mask_sigmoid.shape == torch.Size([1, 4096])
                    masked_intermediate_output_rotated = (
                        1 - l4_mask_sigmoid
                    ) * intermediate_output_rotated
                    masked_vector_source_rotated = (
                        l4_mask_sigmoid * vector_source_rotated
                    )
                    assert (
                        masked_intermediate_output_rotated.shape
                        == masked_vector_source_rotated.shape
                        == torch.Size([self.batch_size, 4096])
                    )

                    iia_vector_rotated = (
                        masked_intermediate_output_rotated
                        + masked_vector_source_rotated
                    )
                    assert iia_vector_rotated.shape == torch.Size(
                        [self.batch_size, 4096]
                    )
                    # TODO: first add them then unrotate.

                    # masked_intermediate_output_unrotated = torch.matmul(masked_intermediate_output_rotated,self.rotate_layer.weight.T)
                    # masked_vector_source_unrotated = torch.matmul(masked_vector_source_rotated,self.rotate_layer.weight.T)

                    iia_vector = torch.matmul(
                        iia_vector_rotated, self.rotate_layer.weight.T
                    )

                    # iia_vector = masked_intermediate_output_unrotated + masked_vector_source_unrotated

                    # intermediate_output = (1 - self.l4_mask) * intermediate_output[:,self.intervened_token_idx,:].unsqueeze(0) + self.l4_mask * vector_source[0][:,self.intervened_token_idx,:].unsqueeze(0)
                    iia_vector = iia_vector.reshape(-1, 1, 4096)
                    assert (
                        (iia_vector).shape
                        == vector_source[0][:, self.intervened_token_idx, :]
                        .unsqueeze(1)
                        .shape
                        == torch.Size([self.batch_size, 1, 4096])
                    )
                    # Create a new tuple with the modified intermediate_output
                    # modified_output = (intermediate_output,) + self.model.transformer.h[self.layer_intervened].output[1:]
                    assert (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0][:, self.intervened_token_idx, :]
                        .shape
                        == iia_vector.squeeze(1).shape
                        == torch.Size([self.batch_size, 4096])
                    )
                    self.model.transformer.h[self.layer_intervened].output[0][
                        :, self.intervened_token_idx, :
                    ] = iia_vector.squeeze(1)
                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = []
            for index in range(intervened_base_output.shape[0]):
                predicted_text.append(
                    self.model.tokenizer.decode(
                        intervened_base_output[index].argmax(dim=-1)
                    ).split()[-1]
                )

            return intervened_base_output, predicted_text

        elif self.method == "sae masking":

            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:
                    source = self.model.transformer.h[self.layer_intervened].output

                with tracer.invoke(base_ids) as runner_:

                    base = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    encoded_base, base_info = self.autoencoder.encode(base)
                    encoded_source, source_info = self.autoencoder.encode(source[0])

                    # Clone the tensors to avoid in-place operations
                    encoded_base_modified = encoded_base.clone()
                    encoded_source_modified = encoded_source.clone()

                    assert base_info == source_info

                    # Apply the mask in a non-inplace way
                    modified_base = encoded_base_modified[
                        :, self.intervened_token_idx, :
                    ] * (1 - l4_mask_sigmoid)
                    modified_source = (
                        encoded_source_modified[:, self.intervened_token_idx, :]
                        * l4_mask_sigmoid
                    )

                    # Assign the modified tensors to the correct indices
                    encoded_base_modified = encoded_base_modified.clone()
                    encoded_source_modified = encoded_source_modified.clone()

                    # Combine the modified tensors
                    new_acts = encoded_base_modified.clone()
                    new_acts[:, self.intervened_token_idx, :] = (
                        modified_base + modified_source
                    )

                    iia_vector = self.autoencoder.decode(new_acts, base_info)

                    # Use a copy to avoid in-place modification
                    h_layer_output_copy = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    h_layer_output_copy[:, self.intervened_token_idx, :] = iia_vector[
                        :, self.intervened_token_idx, :
                    ]

                    # Update the model's output with the modified copy
                    self.model.transformer.h[self.layer_intervened].output[0][
                        :, :, :
                    ] = h_layer_output_copy

                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = []
            for index in range(intervened_base_output.shape[0]):
                predicted_text.append(
                    self.model.tokenizer.decode(
                        intervened_base_output[index].argmax(dim=-1)
                    ).split()[-1]
                )

            return intervened_base_output, predicted_text
