from imports import *
from ravel_data_prep import *
from eval_gpt2 import *

def overlap_measure():
    
    with open("gpt2_comfy_top1_country.json", "r") as file:
        country_data = json.load(file)
    
    with open("gpt2_comfy_top1_continent.json", "r") as file:
        continent_data = json.load(file)
    
    country_cities = []
    for i in country_data["sentences"]:
        country_cities.append(i[0].split(".")[-1].split()[0])
    
    continent_cities = []
    for i in continent_data["sentences"]:
        continent_cities.append(i[0].split(".")[-1].split()[0])
    
    overlap = list(set(country_cities) & set(continent_cities))
    print(overlap)
    print(f"The total number of overlapping words are {len(overlap)}")
    
    return overlap


def model_eval(model, eval_file_path, attribute):
    
    with open(eval_file_path, 'r') as file:
        data = json.load(file)

    comfy_data = []
    correct = 0
    for sample, label in data:
        with model.trace(sample):
            output = model.lm_head.output.argmax(dim = -1).save()
    
        prediction = model.tokenizer.decode(output[0][-1])
        if prediction.split()[0] == label.split()[0]:
            correct+=1

            comfy_data.append([sample, label])
        
    with open(f"comfy_{attribute}_top1.json", "w") as file:
        json.dump(comfy_data, file)
    

    print(f"the accuracy for {args.attribute} is {correct/len(data)}")
    

def intervention_dataset(overlapping_cities):
    
    with open("comfy_country_top1.json", "r") as file:
        country_data = json.load(file)
    
    with open("comfy_continent_top1.json", "r") as file:
        continent_data = json.load(file)
    
    
    def dataset(data, attribute):
        
        new_data = []
        
        for i in data:
        
            city_name = i[0].split(".")[-1].split()[0]
            if city_name in overlapping_cities:
                pass
            else:
                continue
        
            for j in data:
                
                indented_city_name = j[0].split(".")[-1].split()[0]
                if indented_city_name in overlapping_cities:
                    pass
                else:
                    continue
        
                if i != j:
                    if i[1] != j[1]:
                        new_data.append([i,j])
                elif i == j:
                    pass
        
        print(f"The total number of sample pairs in {attribute} are {len(new_data)}")
        
        with open(f"{attribute}_intervention_dataset.json", "w") as file:
            json.dump(new_data, file)

    dataset(country_data, "country")
    dataset(continent_data, "continent")
    


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
    

    '''
    #Now I will have to make the code for taking the accuracy on the prepared selected dataset of ravel

    '''
    
    # model_eval(eval_file_path=args.eval_file_path, model = model, attribute=args.attribute)
    # overlapping_cities = overlap_measure()
    
    # creating the intervention dataset of overlapping cities. 
    # intervention_dataset(overlapping_cities=overlapping_cities)
    

    with open("continent_intervention_dataset.json", "r") as file:
        continent_intervention_data = json.load(file)
    
    '''
    # Now, I will have to make the code for intervention of the data in the first layer of GPT2
    '''
    
    # tokenizer.pad_token = tokenizer.eos_token
    
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
    

    def safe_split(word):
        try:
            a = word.split()[0]
            return a 
        except:
            return " "

    i = 1
    total_samples_processed = 0
    
    for sample_no in tqdm(range(len(continent_intervention_data))):
        
        sample = continent_intervention_data[sample_no]
        base = sample[0][0]
        source = sample[1][0]
        base_label = sample[0][1]
        source_label = sample[1][1]
        # base_city = sample[:][0][0].split(".")[-1].split()[0]
        
        # pprint(base)
        # base_label = sample[:][0][1]
        # base_city = sample[:][0][0].split(".")[-1].split()[0]
        
        # source_text = sample[:][1][0]
        # source_label = sample[:][1][1]

        base_ids = []
        source_ids = []
        
        base_ids = tokenizer.encode(base, return_tensors='pt') 
        base_tokens = tokenizer.tokenize(base)
        source_ids = tokenizer.encode(source, return_tensors='pt')
        source_tokens = tokenizer.tokenize(source) 
        
        base_ids = base_ids.to(DEVICE)
        source_ids = source_ids.to(DEVICE)

        base_ids = base_ids.type(torch.LongTensor)
        source_ids = source_ids.type(torch.LongTensor)
        
        # print(source_tokens)
        
        intervened_token_idx = -1 # -8 for continent and -9 for country
        
        
        # for i in range(0,9):
        
        # only intervening for same shapes as intervening on different shapes misleads the results, giving 0 acc for intervention (done only for initial experimentation)
        if source_ids.shape != base_ids.shape:
            continue
    
        print()
        print(base_tokens)
        print(source_tokens)
        print(f"Source token {intervened_token_idx} : {source_tokens[intervened_token_idx]}, and Base token {intervened_token_idx}: {base_tokens[intervened_token_idx]}")
        
        print(f"The len of source ids is {source_ids.shape} and the base ids is {base_ids.shape}")
        print()

        with model.trace() as tracer:
        
            with tracer.invoke(source_ids) as runner:

                vector_source = model.transformer.h[i].output

            with tracer.invoke(base_ids) as runner_:
                
                print(vector_source.shape)
                model.transformer.h[i].output[0][:,intervened_token_idx,:] = vector_source[0][:,intervened_token_idx,:]
                intervened_base_output = model.lm_head.output.argmax(dim = -1).save()
        
        # intervened_base_output.argamx(dim = -1)[:]
        
        predicted_text = model.tokenizer.decode(intervened_base_output[0][-1])
        # print(f"For Layer {i} we are intervening on the base label '{base_label}' with the source label '{source_label}' and I get the output '{predicted_text}'")
        
        print()
        pprint(f"Base Label: {base_label}")
        pprint(f"Predicted text: {predicted_text}")
        pprint(f"Source Label: {source_label}")
        print()
        
        matches = sum(1 for a, b in zip(predicted_text, source_label) if a == b)
        correct[i].append(matches)
        total_samples_processed+=1
        
    
        if sample_no%100 == 0:
            print(correct[i])
            print(sum(correct[i])/total_samples_processed)
        
    total = len(continent_intervention_data)
    #total = 100

    # for i in range(0,9):
    print(sum(correct[i]))
    print(f"The accuracy of layer {i} is {sum(correct[i])/total}")


    
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

            

    
