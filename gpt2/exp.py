import copy 
import json

with open("final_data_continent.json", "r") as f:
    continent_data = json.load(f)


with open("final_data_country.json", "r") as f:
   country_data = json.load(f)


print(f"len of continent_data", len(continent_data))
print(f"len of country_data", len(country_data))
