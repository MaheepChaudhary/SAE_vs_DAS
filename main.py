from imports import *



def dict_load(activation_dim, 
            dict_embed_path,
            attn_dict_path,
            mlp_dict_path,
            resid_dict_path,
            DEVICE,
            expansion_factor=64):
    
    # dictionary hyperparameters
    dict_id = 10
    expansion_factor = 64
    dictionary_size = expansion_factor * activation_dim
    layer = 4

    submodules = []
    dictionaries = {}

    submodules.append(model.gpt_neox.embed_in)
    dictionaries[model.gpt_neox.embed_in] = AutoEncoder.from_pretrained(
        f'/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/embed/{dict_id}_{dictionary_size}/ae.pt',
        device=DEVICE
    )
    for i in range(layer + 1):
        submodules.append(model.gpt_neox.layers[i].attention)
        dictionaries[model.gpt_neox.layers[i].attention] = AutoEncoder.from_pretrained(
            f'/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/attn_out_layer{i}/{dict_id}_{dictionary_size}/ae.pt',
            device=DEVICE
        )

        submodules.append(model.gpt_neox.layers[i].mlp)
        dictionaries[model.gpt_neox.layers[i].mlp] = AutoEncoder.from_pretrained(
            f'/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/mlp_out_layer{i}/{dict_id}_{dictionary_size}/ae.pt',
            device=DEVICE
        )

        submodules.append(model.gpt_neox.layers[i])
        dictionaries[model.gpt_neox.layers[i]] = AutoEncoder.from_pretrained(
            f'/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/resid_out_layer{i}/{dict_id}_{dictionary_size}/ae.pt',
            device=DEVICE
        )

# metric fn is used to 
def metric_fn(model, labels=None):
    attn_mask = model.input[1]['attention_mask']
    acts = model.gpt_neox.layers[layer].output[0]
    acts = acts * attn_mask[:, :, None]
    acts = acts.sum(1) / attn_mask.sum(1)[:, None]
    
    return t.where(
        labels == 0,
        probe(acts),
        - probe(acts)
    )



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-dict','download_dictionary', type=bool, help="a boolean, helping to get you dictionary if you don't have it.")
    argparser.add_argument('data_dir', type=str, help='directory to save the data')
    argparser.add_argument('-e','--epochs', default=15, type=int, help='number of epochs')
    argparser.add_argument('-lr','--lr', default=0.001, type=float, help='learning rate')
    argparser.add_argument('-btr','batch_size_train', type=int, help='batch size for training')
    argparser.add_argument('-bts','batch_size_test', type=int, help='batch size for testing')
    argparser.add_argument("-d",'device', type=str, help='device to be used')
    argparser.add_argument("-layer",'residual layer', type=str, help="residual layer to be used interevened in the model")
    argparser.add_argument("-activation_dim", type=int, help="activation dimension")
    argparser.add_argument("-ef", "--expansion_factor", default = 64, type=int, help="expansion factor")
    argparser.add_argument("-dict_embed_path", type=str, help="dictionary embedding path")
    argparser.add_argument("-attn_dict_path", type=str, help="attention dictionary path")
    argparser.add_argument("-mlp_dict_path", type=str, help="mlp dictionary path")
    argparser.add_argument("-resid_dict_path", type=str, help="residual dictionary path")
    
    
    args = argparser.parse_args()
    
    model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map=args.device, dispatch=True)
    activation_dim = 512





