from imports import *
from ravel_data_prep import *
from eval_gpt2 import *

def neuron_masking(model, source_ids, base_ids):
    
    with model.trace() as tracer:
        
        with tracer.invoke(source_ids) as runner:

            vector_source = model.transformer.h[layer_intervened].output

        with tracer.invoke(base_ids) as runner_:
            
            # print(vector_source.shape)
            model.transformer.h[layer_intervened].output[0][:,intervened_token_idx,:] = vector_source[0][:,intervened_token_idx,:]
            intervened_base_output = model.lm_head.output.argmax(dim = -1).save()
        
        predicted_text = model.tokenizer.decode(intervened_base_output[0][-1])

    return predicted_text


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path_json", default = "ravel/data/ravel_city_entity_attributes.json", help='Prompting for Ravel Data')
    parser.add_argument("-d", "--device", default = "cuda:1", help='Device to run the model on')
    parser.add_argument("-efp", "--eval_file_path", required = True, help = "file path which you would like to evaluate" )
    parser.add_argument("-m", "--model", default = "gpt2", help= "the model which you would like to evaluate on the ravel dataset")
    parser.add_argument("-a", "--attribute", required = True, help = "name of the attribute on which evaluation is being performned")
    parser.add_argument("-acc", "--accuracy", required=True, help = "type of accuracy of the model on the evaluation dataset, i.e. top 1 or top 5 or top 10")
    parser.add_argument("-tla", "--token_length_allowed", required=True, help = "insert the length you would allow the model to train mask")
    parser.add_argument("-m", "--method", required=True, help="to let know if you want neuron masking, das masking or SAE masking")
    parser.add_argument("-e", "--epochs", default=10, help="# of epochs on which mask is to be trained")

    args = parser.parse_args()
    wandb.init(project="sae_concept_eraser")
    wandb.run.name = f"{args.model}-{args.attribute}-{args.method}-{args.epochs}"
    
    DEVICE = args.device 
    
    # Load gpt2
    if args.model == "gpt2":
        model = LanguageModel("openai-community/gpt2", device_map=DEVICE)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    elif args.model == "mistral":
        model = LanguageModel("mistralai/Mistral-7B-v0.1", device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
    

    with open(args.eval_file_path, "r") as file:
        data = json.load(file)
    
    
    correct = {0:[0],
            1:[0],
            2:[0],
            3:[0],
            4:[0],
            5:[0],
            6:[0],
            7:[0],
            8:[0],
            9:[0],
            10:[0],
            11:[0]}

    layer_intervened = 1 # As the layer has descent performance in the previous metrics of intervention, we will take it.
    intervened_token_idx = -8
    intervention_token_length = args.token_length_allowed
    total_samples_processed = 0


    for sample_no in tqdm(range(len(data))):
        
        # Data Processing
        sample = data[sample_no]
        base = sample[0][0]
        source = sample[1][0]
        base_label = sample[0][1]
        source_label = sample[1][1]
        
        base_ids = tokenizer.encode(base, return_tensors='pt').type(torch.LongTensor).to(DEVICE)
        base_tokens = tokenizer.tokenize(base)
        source_ids = tokenizer.encode(source, return_tensors='pt').type(torch.LongTensor).to(DEVICE)
        source_tokens = tokenizer.tokenize(source) 
        
        # Conditions to filter data:
        
        if len(base_tokens) == args.token_length_allowed:
            pass
        else:
            continue
        
        if source_ids.shape != base_ids.shape:
            continue
        
        assert len(base_tokens) == len(source_tokens)
        token_length = len(base_tokens)
        
        # training the model

        if args.method == "neuron masking":
            predicted_text = neuron_masking(model = model,source_ids=source_ids, base_ids=base_ids)
        elif args.method == "sae masking":
            pass
        elif args.method == "das masking":
            pass
        
        matches = 1 if predicted_text.split()[0] == source_label.split()[0] else 0
        correct[layer_intervened].append(matches)
        total_samples_processed+=1
        

        if sample_no%100 == 0:
            print(sum(correct[layer_intervened]))
            print(sum(correct[layer_intervened])/total_samples_processed)


    print(f"The accuracy of {args.attribute} layer {layer_intervened} is {sum(correct[layer_intervened])/total_samples_processed}")
