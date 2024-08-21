from eval_gpt2 import *
from imports import *
from models import *
from ravel_data_prep import *

random.seed(2)
DEVICE = "mps"


def config(DEVICE):
    model = LanguageModel("openai-community/gpt2", device_map=DEVICE)
    intervened_token_idx = -8
    return model, intervened_token_idx


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--device", default="cuda:1")
    parser.add_argument("-met", "--method", required=True)
    parser.add_argument("-bs", "--batch_size", required=True)
    args = parser.parse_args()

    model, intervened_token_idx = config(args.device)

    model_sae_eval = eval_sae(
        model=model,
        DEVICE=args.device,
        method=args.method,
        intervened_token_idx=intervened_token_idx,
        batch_size=args.batch_size,
    )

    # TODO: Insert the tokenizaton and batching fucntion for the data
    with open("comfy_continent.json", "r") as f:
        contdata = json.load(f)

    with open("comfy_country.json", "r") as f1:
        countdata = json.load(f1)

    print(len(countdata[:, 0]))
#    t_setn = model.tokenizer(contdata[], return_tensors="pt").to(args.device)

# TODO: Insert the inference function to calculate loss and put it in wandb
