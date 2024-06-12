from imports import *
from ravel_data_prep import *
from eval_gpt2 import *

def overlap_measure(country_data, continent_data):
    country_cities = []
    for i in country_data["sentences"]:
        print(i)
        country_cities.append(i[0].split(".")[-1].split()[0])
    
    continent_cities = []
    for i in continent_data["sentences"]:
        print(i)
        continent_cities.append(i[0].split(".")[-1].split()[0])
    
    overlap = list(set(country_cities) & set(continent_cities))
    print(overlap)
    print(len(overlap))

def intervention_dataset(data, attribute):
    new_data = []
    for i in data["sentences"]:
        for j in data["sentences"]:
            if i != j:
                if i[1] != j[1]:
                    new_data.append([i,j])
            elif i == j:
                pass
    
    with open(f"{attribute}_intervention_dataset.json", "w") as file:
        json.dump(new_data, file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path_json", default = "ravel/data/ravel_city_entity_attributes.json", help='Prompting for Ravel Data')
    parser.add_argument("-d", "--device", default = "cuda:1", help='Device to run the model on')
    parser.add_argument("-efp", "--eval_file_path", required = True, help = "file path which you would like to evaluate" )
    parser.add_argument("-m", "--model", default = "gpt2", help= "the model which you would like to evaluate on the ravel dataset")
    parser.add_argument("-a", "--attribute", required = True, help = "name of the attribute on which evaluation is being performned")
    parser.add_argument("-acc", "--accuracy", required=True, help = "type of accuracy of the model on the evaluation dataset, i.e. top 1 or top 5 or top 10")

    args = parser.parse_args()
    # wandb.init(project="sae_concept_eraser")
    # wandb.run.name = f"{args.model}-{args.attribute}"
    
    DEVICE = args.device 
    # DEVICE = torch.device(DEVICE)
    
    # Load gpt2
    if args.model == "gpt2":
        model = LanguageModel("openai-community/gpt2", device_map=DEVICE)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    elif args.model == "mistral":
        model = LanguageModel("mistralai/Mistral-7B-v0.1", device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

    # eval_file_path  = f"/content/{args.attribute}_data.json"
    eval_file_path = args.eval_file_path
    
    with open(eval_file_path, 'r') as file:
        data = json.load(file)

    '''
    Now I will have to make the code for taking the accuracy on the prepared selected dataset of ravel
    '''
    
    # with open("gpt2_comfy_top1_country.json", "r") as file:
    #     country_data = json.load(file)
        
    # with open("gpt2_comfy_top1_continent.json", "r") as file:
    #     continent_data = json.load(file)
    
    # intervention_dataset(country_data, "country")
    # intervention_dataset(continent_data,"continent")
    
    
    
    with open("continent_intervention_dataset.json", "r") as file:
        continent_intervention_data = json.load(file)
    
    '''
    # Now, I will have to make the code for intervention of the data in the first layer of GPT2
    
    '''
    
    tokenizer.pad_token = tokenizer.eos_token
    
    correct = {}
    
    for sample_no in tqdm(range(0,len(continent_intervention_data), 4)):
        
        sample = continent_intervention_data[sample_no:sample_no+4]
        base = [element[0][0] for element in sample]
        source = [element[1][0] for element in sample]
        base_label = [element[0][1] for element in sample]
        source_label = [element[1][1] for element in sample]
        # base_city = sample[:][0][0].split(".")[-1].split()[0]
        
        # pprint(base)
        # base_label = sample[:][0][1]
        # base_city = sample[:][0][0].split(".")[-1].split()[0]
        
        # source_text = sample[:][1][0]
        # source_label = sample[:][1][1]

        base_ids = []
        source_ids = []
        
        base_ids = [tokenizer.encode(sent, return_tensors='pt') for sent in base]
        source_ids = [tokenizer.encode(sent, return_tensors='pt') for sent in source]

        # Pad the sequences manually
        def pad_sequences(sequences, pad_token_id):
            max_length = max(seq.size(1) for seq in sequences)
            padded_sequences = [torch.cat([seq, torch.tensor([[pad_token_id] * (max_length - seq.size(1))])], dim=1) for seq in sequences]
            return torch.cat(padded_sequences, dim=0)

        base_ids = pad_sequences(base_ids, tokenizer.pad_token_id)
        source_ids = pad_sequences(source_ids, tokenizer.pad_token_id)
        
        base_ids = base_ids.to(DEVICE)
        source_ids = source_ids.to(DEVICE)

        base_ids = base_ids.type(torch.LongTensor)
        source_ids = source_ids.type(torch.LongTensor)

        print(f"The shape of the base_ids is {base_ids.shape}")
        print(f"The shape of the source_ids is {source_ids.shape}")
        
        
        # base_tokens = tokenizer.tokenize(base)
        # source_ids = tokenizer.encode(source, return_tensors='pt')
        # source_tokens = tokenizer.tokenize(source)
        
        intervened_token_idx = -9 # -9 is the index of the last word of the city and -10 is the index of the first word of the city
        
        # print(f"The base_token intervened word is {base_tokens[intervened_token_idx]}")
        # print(f"The source_token intervened word is {source_tokens[intervened_token_idx]}")
        

        for i in range(1,11):
        
            
        
            with model.trace() as tracer:
            
                with tracer.invoke(source_ids) as runner:

                    vector_source = model.transformer.h[i].output

                with tracer.invoke(base_ids) as runner_:
                    
                    model.transformer.h[i].output[0][:,50:60,:] = vector_source[0][:,50:60,:]
                    intervened_base_output = model.lm_head.output.save()
            
            predicted_text = [tokenizer.decode(output[-2]) for output in intervened_base_output.argmax(dim = -1)]
            predicted_text = [i.split()[0] for i in predicted_text]
            # print(f"For Layer {i} we are intervening on the base label '{base_label}' with the source label '{source_label}' and I get the output '{predicted_text}'")
            
            print(predicted_text)
            print(source_label)
            
            matches = sum(1 for a, b in zip(predicted_text, source_label) if a == b)
            print(matches)
            print()
        
        # print(f"Accuracy: {correct[1]/200}")
        

    
    total = len(continent_intervention_data)

    for i in range(1,11):
        print(f"The accuracy of layer {i} is {correct[i]/total} for token position {i}")

    
    
    # overlap_measure(country_data=country_data, continent_data=continent_data)
    
    
    # accuracy, correct_arr = eval_on_vanilla_gpt(DEVICE, model, args.model, data, args.attribute, tokenizer, args.accuracy)
    # sentences_json = json.dumps({"sentences": correct_arr}, indent=4)
    
    # with open(f"gpt2_comfy_{args.accuracy}_{args.attribute}.json", "w") as file:
    #     file.write(sentences_json)
    # wandb.log({"Evaluation Accuracy": accuracy})

    # with model.trace() as runner:
    #     with runner.invoke("Aalborg is a city in the continent of") as invoker:
    #         logits = model.lm_head.output.save()

    # probabilities = torch.softmax(logits[:, -1, :], dim=-1)

    # # Get the most likely next token ID
    # next_token_id = torch.argmax(probabilities, dim=-1).item()

    # # Decode the token ID to a string
    # next_token = tokenizer.decode(next_token_id)
    
    # # if next_token == label:
    # #     print("Correct!")
    
    # # elif next_token != label:
    # #     print("Correct Answer: ", label)
    # print("Predicted Answer: ", next_token)
    # #     print("Incorrect!")

            

    
