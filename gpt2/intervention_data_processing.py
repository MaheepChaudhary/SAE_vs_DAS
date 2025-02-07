from imports import *

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
        continent_cities.append(i[0].split(".")[-1].split()[0])
    
    overlap = list(set(country_cities) & set(continent_cities))
    print(overlap)
    print(f"The total number of overlapping words are {len(overlap)}")
    
    return overlap


def intervention_data(model):
    '''
    The file will be able to do the following:
    
    * Merge the country and continent data with a condition that the token length 
    is either 59 and 61 respectively.
    
    * Split the data into three parts: train, validation and test.
    '''
    
    overlapping_cities = overlap_measure()
    
    with open('continent_intervention_dataset.json', 'r') as file:
        continent_data = json.load(file)
    
    with open('country_intervention_dataset.json', 'r') as file:
        country_data = json.load(file)
        
    # all_data = continent_data + country_data
    
    # assert len(all_data) == len(continent_data) + len(country_data)
    
    # random.shuffle(all_data)
    
    def filter(data, token_length_allowed):
        
        new_all_data = []
        
        for sample_no in tqdm(range(len(data))):
            sample = data[sample_no]
            base = sample[0][0]
            source = sample[1][0]
            base_label = sample[0][1]
            source_label = sample[1][1]
            
            base_ids = model.tokenizer.encode(base, return_tensors='pt').type(torch.LongTensor).to(model.device)
            base_tokens = model.tokenizer.tokenize(base)
            source_ids = model.tokenizer.encode(source, return_tensors='pt').type(torch.LongTensor).to(model.device)
            source_tokens = model.tokenizer.tokenize(source) 
            source_label_token = model.tokenizer.tokenize(source_label)
            base_label_token = model.tokenizer.tokenize(base_label)
            
            # The model has the vocab with words with space along side them, so we are making the tokens s.t. they do not split and correspond to their word with integrated space. 
            source_label_mod = " " + source_label.split()[0]
            base_label_mod = " " + base_label.split()[0]
                            
            base_label_ids = model.tokenizer.encode(base_label_mod, return_tensors='pt').squeeze(0).type(torch.LongTensor).to(model.device)
            source_label_ids = model.tokenizer.encode(source_label_mod, return_tensors='pt').squeeze(0).type(torch.LongTensor).to(model.device)
            
            if len(base_tokens) == len(source_tokens) == token_length_allowed and len(source_label_ids) == len(base_label_ids) == 1:
                if base.split(".")[-1].split()[0] in overlapping_cities and source.split(".")[-1].split()[0] in overlapping_cities:
                    new_all_data.append(sample)

        return new_all_data
    
    new_continent_data = filter(continent_data, 61)
    new_country_data = filter(country_data, 59)
    
    with open("filtered_continent_intervention_dataset.json", "w") as file:
        json.dump(new_continent_data, file)
    
    with open("filtered_country_intervention_dataset.json", "w") as file:
        json.dump(new_country_data, file)
    
    print(f"The lenght of original continent data is {len(continent_data)} and the new continent data is {len(new_continent_data)}")
    print(f"The lenght of original country data is {len(country_data)} and the new country data is {len(new_country_data)}")


if __name__ == "__main__":
    model = LanguageModel("openai-community/gpt2", device_map="mps")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    intervention_data(model)