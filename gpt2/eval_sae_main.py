from eval_gpt2 import *
from imports import *

# from models import *
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
    parser.add_argument("-met", "-method", required=True)

    args = parser.parse_args()

    model, intervened_token_idx = config(args.device)

    print(model)
