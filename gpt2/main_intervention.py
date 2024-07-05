from imports import *
from ravel_data_prep import *
from eval_gpt2 import *



def safe_split(word):
    try:
        a = word.split()[0]
        return a 
    except:
        return " "

def overlap_measure():
    
    with open("comfy_country_top1.json", "r") as file:
        country_data = json.load(file)
    
    with open("comfy_continent_top1.json", "r") as file:
        continent_data = json.load(file)
    
    country_cities = []
    for i in country_data:
        country_cities.append(i[0].split(".")[-1].split()[0])
    
    continent_cities = []
    for i in continent_data:
        print(i)
        continent_cities.append(i[0].split(".")[-1].split()[0])
    
    overlap = list(set(country_cities) & set(continent_cities))
    print(overlap)
    print(f"The total number of overlapping words are {len(overlap)}")
    
    return overlap


def model_eval(model, eval_file_path, attribute):
    
    with open("continent_intervention_dataset.json", 'r') as file:
        continent_data = json.load(file)

    with open("country_intervention_dataset.json", 'r') as file:
        country_data = json.load(file)
    
    def filter(data):
        comfy_data = []
        correct = 0
        for index in tqdm(range(len(data))):
            sample, label = data[index]
            with model.trace(sample):
                output = model.lm_head.output.argmax(dim = -1).save()
        
            prediction = model.tokenizer.decode(output[0][-1])
            if prediction.split()[0] == label[-1].split()[0]:
                correct+=1

                comfy_data.append([sample, label])
            
        with open(f"comfy_{attribute}_top1.json", "w") as file:
            json.dump(comfy_data, file)
        
        print(f"the accuracy for {args.attribute} is {correct/len(data)}")

    filter(continent_data)
    filter(country_data)
    

def intervention_dataset(overlapping_cities):
    
    with open("comfy_country_top1.json", "r") as file:
        country_data = json.load(file)
    
    with open("comfy_continent_top1.json", "r") as file:
        continent_data = json.load(file)
    
    
    def dataset(data, attribute):
        
        new_data = []
        
        for i in data:
        
            for j in data:
                
                if i[0].split(".")[-1].split()[0] in overlapping_cities and j[0].split(".")[-1].split()[0] in overlapping_cities:
            
                    if i != j:
                        # if i[1] != j[1]:
                        new_data.append([i,j])
                    elif i == j:
                        pass
                else:
                    pass
        
        print(f"The total number of sample pairs in {attribute} are {len(new_data)}")
        
        with open(f"{attribute}_intervention_dataset.json", "w") as file:
            json.dump(new_data, file)

    dataset(country_data, "country")
    dataset(continent_data, "continent")
    
def data_process(sample, model, attribute):
    
    '''
    This is used to extract data in the passing format of the model, extracting base, source and converting them.
    '''
    
    base = sample[0][0]
    source = sample[1][0]
    base_label = sample[0][1]
    source_label = sample[1][1]
    
    base_ids = tokenizer.encode(base, return_tensors='pt', padding=True, truncation=True).type(torch.LongTensor).to(DEVICE)
    base_tokens = tokenizer.tokenize(base)
    source_ids = tokenizer.encode(source, return_tensors='pt', padding=True, truncation=True).type(torch.LongTensor).to(DEVICE)
    # source_tokens = tokenizer.tokenize(source) 
    
    if source_ids.shape[1] != base_ids.shape[1]:
        return False, None, None, None, None, source, base
    
    if source_ids.shape[1] != allowed_token_length or base_ids.size()[1] != allowed_token_length:
        return False, None, None, None, None, source, base
    
    elif source_ids.shape[1] == base_ids.shape[1] == allowed_token_length:
        return True, base_ids, source_ids, base_label, source_label, source, base    

def intervention(model, source_ids, base_ids, layer_index, intervened_token_idx):
    
    '''
    This is defined to do intervention from the source to the base.
    '''

    with model.generate(max_new_tokens=1, pad_token_id=tokenizer.eos_token_id) as tracer:

        with tracer.invoke(source_ids):

            vector_source = model.transformer.h[layer_index].output

        with tracer.invoke(base_ids):

            model.transformer.h[layer_index].output[0][:,intervened_token_idx,:] = vector_source[0][:,intervened_token_idx,:]
            intervened_base_output = model.lm_head.output.argmax(dim = -1).save()
            # intervened_base_output = model.generator.output.save()

    predicted_text = tokenizer.decode(intervened_base_output[0][-1])

    return predicted_text

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
    allowed_token_length = 59 if args.attribute == "country" else 61    
    DEVICE = args.device 
    
    # Load gpt2
    if args.model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        # tokenizer.padding_side = "left"
        model = LanguageModel("openai-community/gpt2", device_map=DEVICE)
        # tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        
    elif args.model == "mistral":
        model = LanguageModel("mistralai/Mistral-7B-v0.1", device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

    # tokenizer = model.tokenizer
    
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # eval_file_path  = f"/content/{args.attribute}_data.json"
    

    '''
    Now I will have to make the code for taking the accuracy on the prepared selected dataset of ravel
    '''
    
    # model_eval(eval_file_path=args.eval_file_path, model = model, attribute=args.attribute)
    overlapping_cities = overlap_measure()
    
    # creating the intervention dataset of overlapping cities. 
    intervention_dataset(overlapping_cities=overlapping_cities)
    
    
    with open(args.eval_file_path, "r") as file:
        data = json.load(file)
    
    with open("comfy_country_top1.json", "r") as file:
        comfy_country_data = json.load(file)
        comfy_country_cities = [sample.split(".")[-1].split()[0] for sample, label in comfy_country_data]
        
    with open("comfy_continent_top1.json", "r") as file:
        comfy_continent_data = json.load(file)
        comfy_continent_cities = [sample.split(".")[-1].split()[0] for sample, label in comfy_continent_data]
    
    print(f"The total number of samples with which GPT-2 is comfortable for country dataset are {len(comfy_country_data)}")
    print(f"The total length of the cities with which GPT-2 is comfortable for country dataset are {len(set(comfy_country_cities))}")
    print(f"The total number of samples with which GPT-2 is comfortable for continent dataset are {len(comfy_continent_data)}")    
    print(f"The total length of the cities with which GPT-2 is comfortable for continent dataset are {len(set(comfy_continent_cities))}")
    print(f"The total number of intersecting cities between country and continent are {len(set(comfy_country_cities) & set(comfy_continent_cities))}")
        
    correct = {i:[0] for i in range(0,12)}
    
        
    total_samples_processed = 0
    
    if args.attribute == "continent":
        len_correct = {61:0, 62:0, 63:0, 64:0}
        len_correct_total = {61:0, 62:0, 63:0, 64:0}
    
    elif args.attribute == "country":
        len_correct = {59:0, 60:0, 61:0, 62:0}
        len_correct_total = {59:0, 60:0, 61:0, 62:0}

    all_cities = []
    count = 0
    len_arr = {}
    total_samples_processed = 0
    
    for sample_no in tqdm(range(len(data))):
        
        sample = data[sample_no]
        proceed, base_ids, source_ids, base_label, source_label, source, base = data_process(sample, model, args.attribute)
        
        if not proceed: continue 

        assert base_ids.size()[1] == source_ids.size()[1] == allowed_token_length
        intervened_token_idx = -8
        
        # for layer_index in range(0,12):
        for layer_index in range(0,1):
            
            # layer_index = 0
            predicted_text = intervention(model=model, source_ids=source_ids, base_ids=base_ids, layer_index=layer_index, intervened_token_idx=intervened_token_idx)
            
            # print(f"Layer Index: {layer_index}")
            # print(f"Base Label: {base_label}")
            # print(f"Source Label: {source_label}")
            # print(f"Predicted Text: {predicted_text}")
            # print()
            
            len_arr[base_ids.size()[1]] = 1 if base_ids.size()[1] not in len_arr else len_arr[base_ids.size()[1]]+1
            
            # The prediction would be done based on the condition of the length of the source label.
            matches = 1 if predicted_text.split()[0] == source_label.split()[0] else 0
            assert type(safe_split(predicted_text)[0]) == type(safe_split(source_label)[0]) == str
            assert safe_split(predicted_text)[0] and safe_split(source_label)[0] != " "
            
            if matches == 0:
                print(predicted_text.split()[0], source_label.split()[0])
            
            # The correct has all the total number of correct samples in each category of the [layer]. 
            # and len_correct contains the total number of correct samples in each category of [length of tokens].
            correct[layer_index].append(matches); 
            len_correct_total[base_ids.size()[1]]+=1; len_correct[base_ids.size()[1]]+=matches
        
            if total_samples_processed%100 == 0 and total_samples_processed != 0:
                print(correct[layer_index])
                print(sum(correct[layer_index])/total_samples_processed)
        
        total_samples_processed+=1
        
    # wandb.run.name = f"{args.model}-{args.attribute}-ttl_samp_proc{total_samples_processed}"
    
    for layer_index in range(0,1):
        print(f"The accuracy of {args.attribute} layer {layer_index} is {sum(correct[layer_index])/total_samples_processed}")
        
        # wandb.log({"Layer-wise Intervention Accuracy": sum(correct[layer_index])/total_samples_processed})
    
    if args.attribute == "continent":
        # for index in [61,62,63]:
        for index in [61]:
            wandb.log({"Length-wise Intervention Accuracy": len_correct[index]/len_correct_total[index]})
            # print(f"Accuracy of Length {index}: {len_correct[index]/len_correct_total[index]}")
    
    elif args.attribute == "country":
        # for index in [59,60,61]:
        for index in [59]:
            # wandb.log({"Length-wise Intervention Accuracy": len_correct[index]/len_correct_total[index]})
            print(f"Accuracy of Length {index}: {len_correct[index]/len_correct_total[index]}")
