from imports import *
from ravel_data_prep import *
from eval_gpt2 import *
from models import *

def config(file_path, learning_rate, token_length):
    
    # Load gpt2
    if args.model == "gpt2":
        model = LanguageModel("openai-community/gpt2", device_map = DEVICE)

    with open(file_path, "r") as file:
        data = json.load(file)
    

    layer_intervened = 1 # As the layer has descent performance in the previous metrics of intervention, we will take it.
    intervened_token_idx = -8
    intervention_token_length = token_length

    return data, model, layer_intervened, intervened_token_idx

def data_processing(model, samples, token_length_allowed, attribute, DEVICE):
    
    bases = list(np.array(samples)[:,0,0])
    sources = list(np.array(samples)[:,1,0])
    # print(bases)
    base_labels = list(np.array(samples)[:,0,1])
    # print(base_labels)
    source_labels = list(np.array(samples)[:,1,1])
    assert len(bases) == len(sources) == len(base_labels) == len(source_labels) == 32
    
    base_ids = model.tokenizer(bases, padding=True, return_tensors='pt').to(DEVICE)
    source_ids = model.tokenizer(sources, padding=True, return_tensors='pt').to(DEVICE)

    source_tokens = model.tokenizer(sources) 
    base_tokens = model.tokenizer(bases)
    source_label_token = model.tokenizer(source_labels)
    base_label_token = model.tokenizer(base_labels)
    
    # The model has the vocab with words with space along side them, so we are making the tokens s.t. they do not split and correspond to their word with integrated space. 
    source_label_mods = [" " + label.split()[0] for label in source_labels]
    base_label_mods = [" " + label.split()[0] for label in base_labels]
    
    base_label_ids = model.tokenizer(base_label_mods, return_tensors='pt').to(DEVICE)
    source_label_ids = model.tokenizer(source_label_mods, return_tensors='pt').to(DEVICE)
    
    allowed_token_length = 59 if attribute == "country" else 61
    
    # if source_ids.shape[1] == base_ids.shape[1] == allowed_token_length:
    #     return True, base_ids, source_ids, base_label_ids, source_label_ids, source, base   

    # else:
    #     return False, base_ids, source_ids, base_label_ids, source_label_ids, source_label, base_label  

    assert token_length_allowed == 61 if attribute == "continent" else 59
    
    # Conditions to filter data:
    if len(base_tokens) == len(source_tokens) == token_length_allowed and len(source_label_ids) == len(base_label_ids) == 1:
        proceed = True
        assert len(base_tokens) == len(source_tokens) == token_length_allowed
    else:
        proceed = False

    proceed = True
    return proceed, base_ids, source_ids, base_label_ids, source_label_ids, source_labels, base_labels

def train_data_processing():
    
    with open("filtered_continent_intervention_dataset.json", "r") as file:
        continent_data = json.load(file)
    
    with open("filtered_country_intervention_dataset.json", "r") as file:
        country_data = json.load(file)
    
    data = continent_data + country_data
    random.shuffle(data)
    # print(data)
    
    train_data = data[:int(0.7*len(data))]
    val_data = data[int(0.7*len(data)):int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]
    return train_data, val_data, test_data

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path_json", default = "ravel/data/ravel_city_entity_attributes.json", help='Prompting for Ravel Data')
    parser.add_argument("-d", "--device", default = "cuda:1", help='Device to run the model on')
    parser.add_argument("-efp", "--eval_file_path", required = True, help = "file path which you would like to evaluate" )
    parser.add_argument("-m", "--model", default = "gpt2", help= "the model which you would like to evaluate on the ravel dataset")
    parser.add_argument("-a", "--attribute", required = True, help = "name of the attribute on which evaluation is being performned")
    # parser.add_argument("-acc", "--accuracy", required=True, help = "type of accuracy of the model on the evaluation dataset, i.e. top 1 or top 5 or top 10")
    parser.add_argument("-tla", "--token_length_allowed", required=True, type = int, help = "insert the length you would allow the model to train mask")
    parser.add_argument("-method", "--method", required=True, help="to let know if you want neuron masking, das masking or SAE masking")
    parser.add_argument("-e", "--epochs", default=1, type = int, help="# of epochs on which mask is to be trained")
    parser.add_argument("-ef", "--expansion_factor", default=1, help="expansion factor for SAE")
    parser.add_argument("-lr", "--learning_rate", default=0.01, help="learning rate for the optimizer")
    parser.add_argument("-t", "--task", required=True, help="task to perform, i.e. train or test")
    parser.add_argument("-svd", "--saved_model_path", default="gpt2/models/saved_model.pth", help="path to the saved model")
    parser.add_argument("-n", "--notes", default="", help = "Any notes you want to write for the wandb graph")

    args = parser.parse_args()
    # wandb.init(project="sae_concept_eraser")
    # wandb.run.name = f"{args.method}-{args.epochs}-{args.notes}"
    DEVICE = args.device 

    data, model, layer_intervened, intervened_token_idx, = config(file_path = args.eval_file_path, learning_rate = args.learning_rate,
                                                                                token_length = args.token_length_allowed)
    # model.to(DEVICE)
    training_model = my_model(model = model, DEVICE=DEVICE, method=args.method, token_length_allowed=args.token_length_allowed, expansion_factor=args.expansion_factor,
                            layer_intervened=layer_intervened, intervened_token_idx=intervened_token_idx)

    training_model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    
    for name, param in training_model.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}')
    optimizer = optim.Adam(training_model.parameters(), lr=args.learning_rate)

    train_data, val_data, test_data = train_data_processing()

    #Inserting the temperature
    total_step = 0
    # target_total_step = len(batches) * args.epochs
    #TODO: The total number of batches is total_no_samples/batch_len
    batch_size = 1
    target_total_step = int(len(train_data)/(batch_size) * args.epochs)
    temperature_start = 10.0
    temperature_end = 0.1
    temperature_schedule = (
        t.linspace(temperature_start, temperature_end, target_total_step)
        .to(t.bfloat16)
        .to(DEVICE)
    )
    
    temp_idx = 0
    
    if args.task == "train":
    

        for epoch in range(args.epochs):

            correct = {i:[] for i in range(0,12)}
            total_samples_processed = 0
            total_loss = 0.0

            # for sample_no in tqdm(range(len(data))):
            i = 0
            batch_size = 32
            matches = 0
            for sample_no in tqdm(range(int(len(train_data)/batch_size))):
                
                samples = train_data[i*batch_size:i*batch_size+batch_size]
                
                # Data Processing
                proceed, base_ids, source_ids, base_label_ids, source_label_ids, source_label, base_label = data_processing(model = model,
                                                                                                                            samples = samples, 
                                                                                                                            token_length_allowed=args.token_length_allowed, 
                                                                                                                            attribute=args.attribute,
                                                                                                                            DEVICE=DEVICE)
                
                if not proceed: continue
                
                # training the model
                optimizer.zero_grad()  
                
                temperature = temperature_schedule[temp_idx]
                intervened_base_output, predicted_text = training_model(source_ids, base_ids, temperature)
                ground_truth_token_id = source_label_ids
                # ground_truth_token_id = base_label_ids
                vocab_size = model.tokenizer.vocab_size
                ground_truth_one_hot = F.one_hot(ground_truth_token_id["input_ids"], num_classes=vocab_size).float()
                cloned_intervened_base_output = intervened_base_output.clone()
                last_token_output = cloned_intervened_base_output[:,-1,:]
                assert ground_truth_one_hot.squeeze(1).shape == last_token_output.shape
                ground_truth_indices = torch.argmax(ground_truth_one_hot.squeeze(1), dim=1)
                ground_truth_indices = ground_truth_indices.float()
                loss = loss_fn(last_token_output, ground_truth_indices)
                # loss = loss_fn(predicted_logit.view(-1, predicted_logit.size(-1)), ground_truth_token_id.view(-1))
                total_loss += loss.item()
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                predicted_text = [word.split()[0] for word in predicted_text]
                source_label = [word.split()[0] for word in source_label]
                # base_label = [word.split()[0] for word in base_label]
                matches_arr = [i for i in range(len(predicted_text)) if predicted_text[i] == source_label[i]]
                # print(predicted_text)
                # print(source_label)
                # print(matches_arr)
                matches+=len(matches_arr)
                # matches = 1 if predicted_text.split()[0] == source_label.split()[0] else 0
                # correct[layer_intervened].append(matches)
                total_samples_processed +=32
                
                # if sample_no % 100 == 0 and sample_no != 0:
                print(f"Epoch: {epoch}, Sample: {sample_no}, Accuracy: {matches / total_samples_processed:.4f}, Loss: {total_loss / total_samples_processed:.4f}")
                    # wandb.log({"GPT-2 Token Sub-Space Intervention Accuracy": sum(correct[layer_intervened]) / total_samples_processed, "GPT-2 Token Sub-Space Intervention Loss": total_loss / total_samples_processed})
                temp_idx += 1
                i+=1
            
            # Log accuracy and loss to wandb
            epoch_accuracy = sum(correct[layer_intervened]) / total_samples_processed
            print(f"The total samples proceesed for {args.attribute} is {total_samples_processed}")
            # wandb.log({"GPT-2 Sub-Space IIA": epoch_accuracy, "Loss": total_loss / total_samples_processed})

            print(f"Epoch {epoch} finished with accuracy {epoch_accuracy:.4f} and average loss {total_loss / total_samples_processed:.4f}")

            # Validation Data Evaluation
            '''
            with torch.no_grad():
                total_val_samples_processed = 0
                total_val_loss = 0
                
                correct_val = {i:[] for i in range(0,12)}
                for sample_no in range(len(val_data)):
                    sample = val_data[sample_no]
                    # Data Processing
                    proceed, base_ids, source_ids, base_label_ids, source_label_ids, source_label, base_label = data_processing(model = model,
                                                                                                                    samples = samples, 
                                                                                                                    token_length_allowed=args.token_length_allowed, 
                                                                                                                    attribute=args.attribute,
                                                                                                                    DEVICE=DEVICE)
                    
                    if not proceed:
                        continue
                    
                    try:
                        temperature = temperature_schedule[temp_idx]
                    except:
                        temperature = temperature_schedule[temp_idx-1]
                        
                    if args.method == "neuron masking" or args.method == "vanilla" or args.method == "das masking":
                        intervened_base_output, predicted_text = training_model(source_ids, base_ids, temperature)
                    elif args.method == "sae masking":
                        pass
                    
                    ground_truth_token_id = source_label_ids
                    predicted_logit = intervened_base_output[:, -1, :]
                    
                    loss = loss_fn(predicted_logit.view(-1, predicted_logit.size(-1)), ground_truth_token_id.view(-1))
                    total_val_loss += loss.item()
                    
                    # Calculate accuracy
                    matches = 1 if predicted_text.split()[0] == source_label.split()[0] else 0
                    correct_val[layer_intervened].append(matches)
                    total_val_samples_processed += 1
                    
                epoch_val_accuracy = sum(correct_val[layer_intervened]) / total_val_samples_processed
                wandb.log({"Validation Accuracy": epoch_val_accuracy, "Validation Loss": total_val_loss / total_val_samples_processed})
                
                print(f"Epoch {epoch} finished with validation accuracy {epoch_val_accuracy:.4f} and average validation loss {total_val_loss / total_val_samples_processed:.4f}")
        '''
        # Save the model
        torch.save(training_model.state_dict(), f"models/saved_model_{args.method}_{args.token_length_allowed}_{args.attribute}_{args.model}_{args.epochs}.pth")
        
    elif args.task == "test":
        # Load the saved model
        model_path = args.saved_model_path
        model.load_state_dict(torch.load(model_path))
        model.eval()

        correct_test = {i: [] for i in range(0, 12)}
        total_test_samples_processed = 0
        total_test_loss = 0.0

        with torch.no_grad():
            for sample_no in range(len(test_data)):
                sample = test_data[sample_no]
                # Data Processing
                proceed, base_ids, source_ids, base_label_ids, source_label_ids, source_label = data_processing(sample, 
                                                                                                                args.token_length_allowed, 
                                                                                                                args.attribute)
                
                if not proceed:
                    continue
                
                temperature = temperature_schedule[temp_idx]
                if args.method == "neuron masking" or args.method == "vanilla":
                    intervened_base_output, predicted_text = training_model(source_ids, base_ids, temperature)
                elif args.method == "sae masking":
                    pass
                elif args.method == "das masking":
                    pass
                
                ground_truth_token_id = source_label_ids
                predicted_logit = intervened_base_output[:, -1, :]
                
                loss = loss_fn(predicted_logit.view(-1, predicted_logit.size(-1)), ground_truth_token_id.view(-1))
                total_test_loss += loss.item()
                
                # Calculate accuracy
                matches = 1 if predicted_text.split()[0] == source_label.split()[0] else 0
                correct_test[layer_intervened].append(matches)
                total_test_samples_processed += 1
                
                if sample_no % 100 == 0:
                    print(f"Sample: {sample_no}, Test Accuracy: {sum(correct_test[layer_intervened]) / total_test_samples_processed:.4f}, Test Loss: {total_test_loss / total_test_samples_processed:.4f}")

        # Calculate final test accuracy and loss
        test_accuracy = sum(correct_test[layer_intervened]) / total_test_samples_processed
        test_loss = total_test_loss / total_test_samples_processed

        print(f"Test finished with accuracy {test_accuracy:.4f} and average loss {test_loss:.4f}")

        # Log test accuracy and loss to wandb
        wandb.log({"GPT-2 Subspace Test Accuracy": test_accuracy, "Test Loss": test_loss})
