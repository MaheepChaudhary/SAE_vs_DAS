from imports import *
from ravel_data_prep import *
from eval_gpt2 import *
from models import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path_json", default = "ravel/data/ravel_city_entity_attributes.json", help='Prompting for Ravel Data')
    parser.add_argument("-d", "--device", default = "cuda:1", help='Device to run the model on')
    parser.add_argument("-efp", "--eval_file_path", required = True, help = "file path which you would like to evaluate" )
    parser.add_argument("-m", "--model", default = "gpt2", help= "the model which you would like to evaluate on the ravel dataset")
    parser.add_argument("-a", "--attribute", required = True, help = "name of the attribute on which evaluation is being performned")
    parser.add_argument("-acc", "--accuracy", required=True, help = "type of accuracy of the model on the evaluation dataset, i.e. top 1 or top 5 or top 10")
    parser.add_argument("-tla", "--token_length_allowed", required=True, help = "insert the length you would allow the model to train mask")
    parser.add_argument("-method", "--method", required=True, help="to let know if you want neuron masking, das masking or SAE masking")
    parser.add_argument("-e", "--epochs", default=10, help="# of epochs on which mask is to be trained")
    parser.add_argument("-ef", "--expansion_factor", default=1, help="expansion factor for SAE")

    args = parser.parse_args()
    wandb.init(project="sae_concept_eraser")
    wandb.run.name = f"{args.model}-TLA_{args.token_length_allowed}-{args.attribute}-{args.method}-{args.epochs}"
    
    DEVICE = args.device 
    
    # Load gpt2
    if args.model == "gpt2":
        model = LanguageModel("openai-community/gpt2", device_map=DEVICE)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    elif args.model == "mistral":
        model = LanguageModel("mistralai/Mistral-7B-v0.1", device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
    

    with open(args.eval_file_path, "r") as file:
        data = json.load(file)
    

    layer_intervened = 1 # As the layer has descent performance in the previous metrics of intervention, we will take it.
    intervened_token_idx = -8
    intervention_token_length = args.token_length_allowed

    training_model = my_model(model = model, DEVICE=DEVICE, method=args.method, token_length_allowed=args.token_length_allowed, expansion_factor=args.expansion_factor,
                            layer_intervened=layer_intervened, intervened_token_idx=intervened_token_idx)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(training_model.parameters(), lr=args.learning_rate)


    for epoch in args.epochs:

        correct = []
        total_samples_processed = 0
        total_loss = 0.0
        
        for sample_no in tqdm(range(len(data))):
            
            # Data Processing
            sample = data[sample_no]
            base = sample[0][0]
            source = sample[1][0]
            base_label = sample[0][1]
            source_label = sample[1][1]
            
            base_ids = tokenizer.encode(base, return_tensors='pt').type(torch.LongTensor).to(DEVICE)
            base_tokens = tokenizer.tokenize(base)
            source_ids = tokenizer.encode(source, return_tensors='pt').type(torch.LongTensor).to(DEVICE)
            source_tokens = tokenizer.tokenize(source) 
            
            # Conditions to filter data:
            
            if len(base_tokens) == args.token_length_allowed:
                pass
            else:
                continue
            
            if source_ids.shape != base_ids.shape:
                continue
            
            assert len(base_tokens) == len(source_tokens)
            token_length = len(base_tokens)
            
            # training the model
            optimizer.zero_grad()  # Reset gradients
            
            # training the model
            if args.method == "neuron masking":
                predicted_text = training_model(source_ids, base_ids, layer_intervened, intervened_token_idx)
            elif args.method == "sae masking":
                pass
            elif args.method == "das masking":
                pass
            
            predicted_ids = tokenizer.encode(predicted_text, return_tensors='pt').type(torch.LongTensor).to(DEVICE)
            source_label_ids = tokenizer.encode(source_label, return_tensors='pt').type(torch.LongTensor).to(DEVICE)
            
            if predicted_ids.shape != source_label_ids.shape:
                continue
            
            loss = loss_fn(predicted_ids, source_label_ids)
            total_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            matches = 1 if predicted_text.split()[0] == source_label.split()[0] else 0
            correct[layer_intervened].append(matches)
            total_samples_processed += 1
            
            if sample_no % 100 == 0:
                print(f"Epoch: {epoch}, Sample: {sample_no}, Accuracy: {sum(correct[layer_intervened]) / total_samples_processed:.4f}, Loss: {total_loss / total_samples_processed:.4f}")
        
        # Log accuracy and loss to wandb
        epoch_accuracy = sum(correct[layer_intervened]) / total_samples_processed
        wandb.log({"Epoch": epoch, "GPT-2 Accuracy": epoch_accuracy, "Loss": total_loss / total_samples_processed})

        print(f"Epoch {epoch} finished with accuracy {epoch_accuracy:.4f} and average loss {total_loss / total_samples_processed:.4f}")

            
