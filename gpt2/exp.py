# import copy
# import json
#
# from e2e_sae.e2e_sae import SAETransformer
#
# with open("final_data_continent.json", "r") as f:
#     continent_data = json.load(f)
#
#
# with open("final_data_country.json", "r") as f:
#     country_data = json.load(f)
#
#
# print(f"len of continent_data", len(continent_data))
# print(f"len of country_data", len(country_data))
#

from e2e_sae.e2e_sae import SAETransformer

model = SAETransformer.from_wandb("sparsify/gpt2/xomqkliv")

print(model.saes)
# or, if stored locally
# model = SAETransformer.from_local_path("/path/to/checkpoint/dir")
