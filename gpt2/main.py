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
                new_data.append([i,j])
            elif i == j:
                pass
    
    with open(f"{attribute}_intervention_dataset.json", "w") as file:
        json.dump(new_data, file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path_json", default = "ravel/data/ravel_city_entity_attributes.json", help='Prompting for Ravel Data')
    parser.add_argument("-d", "--device", default = "mps", help='Device to run the model on')
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
    
    # with open(eval_file_path, 'r') as file:
    #     data = json.load(file)

    '''
    Now I will have to make the code for taking the accuracy on the prepared selected dataset of ravel
    '''
    
    with open("gpt2_comfy_top1_country.json", "r") as file:
        country_data = json.load(file)
        
    with open("gpt2_comfy_top1_continent.json", "r") as file:
        continent_data = json.load(file)
    
    # intervention_dataset(country_data, "country")
    # intervention_dataset(continent_data,"continent")
    
    with open("continent_intervention_dataset.json", "r") as file:
        continent_intervention_data = json.load(file)
    
    # pprint(len(continent_intervention_data))
    
    '''
    Now, I will have to make the code for intervention of the data in the first layer of GPT2
    
    '''
    
    text = [['Toronto is a city in the continent of North America. Beijing is a city in '
            'the continent of Asia. Miami is a city in the continent of North America. '
            'Santiago is a city in the continent of South America. London is a city in '
            'the continent of Europe. Anyang is a city in the continent of ',
            'Asia'],
            ['Toronto is a city in the continent of North America. Beijing is a city in '
            'the continent of Asia. Miami is a city in the continent of North America. '
            'Santiago is a city in the continent of South America. London is a city in '
            'the continent of Europe. Makamba is a city in the continent of ',
            'Africa']]
    
    base = text[0][0]
    base_label = text[0][1]
    base_city = text[0][0].split(".")[-1].split()[0]
    
    source_text = text[1][0]
    source_label = text[1][1]

    input_ids = tokenizer.encode(base, return_tensors='pt')

    # Identify the position of the word in the input sequence
    # tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # word_position = tokens.index(base_city)

    print(model)
    
    print(f"the lenght of the text is {len(base)}")
    with model.trace(input_ids):
        vector = model.transformer.h[0].mlp.output.save()

        # model.transformer.h[0].ln_1.output[:,word_position,:] = torch.zeros(vector.shape)
        
        logits = model.lm_head.output.save()
        
    print(f"The shape of the vector is {vector.shape}")
    predicted_text = model.tokenizer.decode(logits.argmax(dim = -1)[0][-2])
    print(predicted_text)
    print(source_label)
    
    # overlap_measure(country_data=country_data, continent_data=continent_data)
    
    

    # accuracy, correct_arr = eval_on_vanilla_gpt(DEVICE, model, args.model, data["sentences"], args.attribute, tokenizer, args.accuracy)
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

            

    