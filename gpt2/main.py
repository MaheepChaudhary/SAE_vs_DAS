from imports import *
from ravel_data_prep import *
from eval_gpt2 import *

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
    
    with open(eval_file_path, 'r') as file:
        data = json.load(file)
    
    accuracy = eval_on_vanilla_gpt(DEVICE, model, args.model, data, args.attribute, tokenizer, args.accuracy)
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

            

    