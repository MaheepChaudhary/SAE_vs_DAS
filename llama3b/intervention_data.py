from imports import *

def analyse(model, data, attribute):
    new_data = []
    for sample_no in tqdm(range(len(data))):
        sample = data[sample_no]
        if len(model.tokenizer.tokenize(sample[0])) == 61 and attribute == "continent":
            with model.trace(sample[0]) as tracer:
                output = model.lm_head.output[0].save()
        
            prediction = model.tokenizer.decode(output.argmax(dim = -1)[-1]).split()[0]
            city = sample[0].split()[-8]
            if prediction == sample[1]:
                new_data.append([sample[0], sample[1]])

        if len(model.tokenizer.tokenize(sample[0])) == 59 and attribute == "country":

            with model.trace(sample[0]) as tracer:
                output = model.lm_head.output[0].save()
        
            prediction = model.tokenizer.decode(output.argmax(dim = -1)[-1]).split()[0]
            city = sample[0].split()[-8]
            if prediction == sample[1]:
                new_data.append([sample[0], sample[1]])

        else:
            continue
    
    with open(f"comfy_{attribute}.json", "w") as f:
        json.dump(new_data, f)




def overlap_measure(country_data, continent_data):

    list_of_country_cities = []
    list_of_continent_cities = []

    for sample in country_data:
        list_of_country_cities.append(sample[0].split()[-8])

    for sample in continent_data:
        list_of_continent_cities.append(sample[0].split()[-8])

    ovelapping_cities = set(list_of_country_cities) & set(list_of_continent_cities)

    print(f"The total number of overlapping cities are {len(overlapping_cities)}")




if __name__ == "__main__":  

    model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map = t.device("cuda:1"))
    print(model)
    
    with open("vanilla_data/continent_data.json", "r") as f:
        continent_data = json.load(f)


    with open("vanilla_data/country_data.json", "r") as f:
        country_data = json.load(f)


    analyse(model, continent_data,"continent")
    analyse(model, country_data, "country")



