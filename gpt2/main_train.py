from imports import *
from ravel_data_prep import *
from eval_gpt2 import *
from models import *

def config(learning_rate, token_length):
    
    model = LanguageModel("openai-community/gpt2", device_map = DEVICE)

    intervened_token_idx = -8
    intervention_token_length = token_length

    return model, intervened_token_idx

def data_processing(model, samples, token_length_allowed, attribute, DEVICE, batch_size):
    
    # print(np.array(samples).shape)
    bases = list(np.array(samples)[:,0,0])
    sources = list(np.array(samples)[:,1,0])
    base_labels = list(np.array(samples)[:,0,1])
    source_labels = list(np.array(samples)[:,1,1])
    assert len(bases) == len(sources) == len(base_labels) == len(source_labels) == batch_size
    
    base_ids = model.tokenizer(bases, return_tensors='pt').to(DEVICE)
    source_ids = model.tokenizer(sources, return_tensors='pt').to(DEVICE)
    # print(base_ids["input_ids"].shape, source_ids["input_ids"].shape)
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

    proceed = True
    return proceed, base_ids, source_ids, base_label_ids, source_label_ids, source_labels, base_labels

def train_data_processing(task, intervention_divided_data, batch_size):
    
    with open("final_data_continent.json", "r") as file:
        continent_data = json.load(file)
    
    with open("final_data_country.json", "r") as file:
        country_data = json.load(file)
    
    random.shuffle(country_data) 
    random.shuffle(continent_data)
    
    if task == "train" or task == "test":
        if intervention_divided_data == "continent":
            data1 = country_data
            data2 = continent_data
        if intervention_divided_data == "country":
            data1 = continent_data
            data2 = country_data
            
        for sample_no in range(len(data1)):
            sample = data1[sample_no]
            base = sample[0][0]
            source = sample[1][0]
            base_label = sample[0][1]
            source_label = sample[1][1]
            
            data1[sample_no][1][1] = base_label
                
        data1_num_batches = np.array(data1).shape[0] // batch_size
        data2_num_batches = np.array(data2).shape[0] // batch_size
        data1_batch_data = [data1[i*batch_size:(i+1)*batch_size] for i in range(data1_num_batches)]
        data2_batch_data = [data2[i*batch_size:(i+1)*batch_size] for i in range(data2_num_batches)]
        assert np.array(data1_batch_data).shape == (data1_num_batches, batch_size, 2, 2)
        assert np.array(data2_batch_data).shape == (data2_num_batches, batch_size, 2, 2)
        data = data1_batch_data + data2_batch_data

        if intervention_divided_data == "continent":
            country_batch_data = data1_batch_data
            continent_batch_data = data2_batch_data
        elif intervention_divided_data == "country":
            continent_batch_data = data1_batch_data
            country_batch_data = data2_batch_data

        
    elif task == "total_iia_train":
        
        
        country_num_batches = np.array(country_data).shape[0] // batch_size
        continent_num_batches = np.array(continent_data).shape[0] // batch_size
        country_batch_data = [country_data[i*batch_size:(i+1)*batch_size] for i in range(country_num_batches)]
        continent_batch_data = [continent_data[i*batch_size:(i+1)*batch_size] for i in range(continent_num_batches)]
        
        assert np.array(country_batch_data).shape == (country_num_batches,batch_size,2,2)
        assert np.array(continent_batch_data).shape == (continent_num_batches,batch_size, 2, 2)
    
        data = country_batch_data + continent_batch_data
    
    random.shuffle(data)
    
    train_data = data[:int(0.7*len(data))]
    val_data = data[int(0.7*len(data)):int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]
    return country_batch_data, continent_batch_data, train_data, val_data, test_data


def train(continent_data, country_data, training_model, model, train_data, optimizer, loss_fn, epochs, token_length_allowed, attribute, temperature_schedule, batch_size, DEVICE, wndb):
    training_model.train()
    
    temp_idx = 0

    for epoch in tqdm(range(epochs)):

        correct = {i:[] for i in range(0,12)}
        total_samples_processed = 0
        total_loss = 0.0

        # for sample_no in tqdm(range(len(data))):
        i = 0
        matches = 0
        for sample_no in range(np.array(train_data).shape[0]):
            
            samples = train_data[sample_no]
            assert np.array(samples).shape == (batch_size, 2, 2)
            # samples = train_data[i*batch_size:(i+1)*batch_size]
            
            # Data Processing
            proceed, base_ids, source_ids, base_label_ids, source_label_ids, source_label, base_label = data_processing(model = model,
                                                                                                                        samples = samples, 
                                                                                                                        token_length_allowed=token_length_allowed, 
                                                                                                                        attribute=attribute,
                                                                                                                        DEVICE=DEVICE,
                                                                                                                        batch_size=batch_size)
            
            if not proceed: continue
            
            # training the model
            optimizer.zero_grad()  
            
            temperature = temperature_schedule[temp_idx]

            intervened_base_output, predicted_text = training_model(source_ids, base_ids, temperature)
            ground_truth_token_id = source_label_ids
            # ground_truth_token_id = base_label_ids
            vocab_size = model.tokenizer.vocab_size
            ground_truth_one_hot = F.one_hot(ground_truth_token_id["input_ids"], num_classes=vocab_size)
            ground_truth_one_hot = ground_truth_one_hot.to(dtype=torch.long)
            last_token_output = intervened_base_output[:,-1,:]
            assert ground_truth_one_hot.squeeze(1).shape == last_token_output.shape
            ground_truth_indices = torch.argmax(ground_truth_one_hot.squeeze(1), dim=1)
            ground_truth_indices = ground_truth_indices.to(dtype=torch.long)
            loss = loss_fn(last_token_output, ground_truth_indices)
            # loss = loss_fn(predicted_logit.view(-1, predicted_logit.size(-1)), ground_truth_token_id.view(-1))
            total_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predicted_text = [word.split()[0] for word in predicted_text]
            source_label = [word.split()[0] for word in source_label]
            matches_arr = [i for i in range(len(predicted_text)) if predicted_text[i] == source_label[i]]
            total_samples_processed +=batch_size
            temp_idx += 1
            i+=1
            # if sample_no % \100 == 0 and sample_no != 0:
            # print(f"Epoch: {epoch}, Sample: {sample_no}, Accuracy: {matches / total_samples_processed:.4f}, Loss: {total_loss / total_samples_processed:.4f}")
        if wndb == "True":
            wandb.log({"GPT-2 Token Sub-Space Intervention Accuracy": matches / total_samples_processed, "GPT-2 Token Sub-Space Intervention Loss": total_loss / total_samples_processed})
        print(f"Epoch: {epoch}, Accuracy: {matches / total_samples_processed:.4f}, Loss: {total_loss / total_samples_processed:.4f}")
        val(training_model, model, val_data, loss_fn, batch_size, token_length_allowed, attribute, temperature, DEVICE, wndb)    
        continent_acc = calculate_accuracy(training_model, model, continent_data, token_length_allowed, attribute, batch_size, DEVICE, temperature)
        country_acc = calculate_accuracy(training_model, model, country_data, token_length_allowed, attribute, batch_size, DEVICE, temperature)
        print(f"Continent Accuracy: {continent_acc}, Country Accuracy: {country_acc}")
        if wndb == "True":
            wandb.log({"Continent Accuracy": continent_acc, "Country Accuracy": country_acc})
        # Log accuracy and loss to wandb
        # epoch_accuracy = matches / total_samples_processed

        # print(f"Epoch {epoch} finished with accuracy {epoch_accuracy:.4f} and average loss {total_loss / total_samples_processed:.4f}")

def calculate_accuracy(training_model, model, data, token_length_allowed, attribute, batch_size, DEVICE, temperature):
    correct_predictions = 0
    total_predictions = 0
    total_samples_processed = 0
    matches = 0
    with t.no_grad():
        for sample_no in range(np.array(data).shape[0]):
            
            samples = data[sample_no]
            assert np.array(samples).shape == (batch_size, 2, 2)
            # samples = train_data[i*batch_size:(i+1)*batch_size]
            
            # Data Processing
            proceed, base_ids, source_ids, base_label_ids, source_label_ids, source_label, base_label = data_processing(model = model,
                                                                                                                        samples = samples, 
                                                                                                                        token_length_allowed=token_length_allowed, 
                                                                                                                        attribute=attribute,
                                                                                                                        DEVICE=DEVICE,
                                                                                                                        batch_size=batch_size)
            
            if not proceed: continue 
            
            intervened_base_output, predicted_text = training_model(source_ids, base_ids, temperature)
            ground_truth_token_id = source_label_ids
            vocab_size = model.tokenizer.vocab_size
            ground_truth_one_hot = F.one_hot(ground_truth_token_id["input_ids"], num_classes=vocab_size)
            ground_truth_one_hot = ground_truth_one_hot.to(dtype=torch.long)
            last_token_output = intervened_base_output[:,-1,:]
            assert ground_truth_one_hot.squeeze(1).shape == last_token_output.shape
            ground_truth_indices = torch.argmax(ground_truth_one_hot.squeeze(1), dim=1)
            ground_truth_indices = ground_truth_indices.to(dtype=torch.long)
            loss = loss_fn(last_token_output, ground_truth_indices)
            
            # Calculate accuracy
            predicted_text = [word.split()[0] for word in predicted_text]
            source_label = [word.split()[0] for word in source_label]
            matches_arr = [i for i in range(len(predicted_text)) if predicted_text[i] == source_label[i]]
            matches+=len(matches_arr)
            total_samples_processed +=batch_size
            
    return matches / total_samples_processed

def val(training_model, model, val_data, loss_fn, batch_size, token_length_allowed, attribute, temperature, DEVICE, wndb):
    with torch.no_grad():
        matches_val = 0
        total_val_samples_processed = 0
        total_val_loss = 0
        
        correct_val = {i:[] for i in range(0,12)}
        for sample_no in range(np.array(val_data).shape[0]):
            samples = val_data[sample_no]
            assert np.array(samples).shape == (batch_size, 2, 2)
            # Data Processing
            proceed, base_ids, source_ids, base_label_ids, source_label_ids, source_label, base_label = data_processing(model = model,
                                                                                                                        samples = samples, 
                                                                                                                        token_length_allowed=token_length_allowed, 
                                                                                                                        attribute=attribute,
                                                                                                                        DEVICE=DEVICE,
                                                                                                                        batch_size=batch_size)
            
            if not proceed:
                continue
                
            intervened_base_output, predicted_text = training_model(source_ids, base_ids, temperature)
            
            ground_truth_token_id = source_label_ids
            vocab_size = model.tokenizer.vocab_size
            ground_truth_one_hot = F.one_hot(ground_truth_token_id["input_ids"], num_classes=vocab_size)
            ground_truth_one_hot = ground_truth_one_hot.to(dtype=torch.long)
            last_token_output = intervened_base_output[:,-1,:]
            assert ground_truth_one_hot.squeeze(1).shape == last_token_output.shape
            ground_truth_indices = torch.argmax(ground_truth_one_hot.squeeze(1), dim=1)
            ground_truth_indices = ground_truth_indices.to(dtype=torch.long)
            loss = loss_fn(last_token_output, ground_truth_indices)
            total_val_loss += loss.item()
            
            # Calculate accuracy
            predicted_text = [word.split()[0] for word in predicted_text]
            source_label = [word.split()[0] for word in source_label]
            matches_arr = [i for i in range(len(predicted_text)) if predicted_text[i] == source_label[i]]
            matches_val+=len(matches_arr)
            total_val_samples_processed +=batch_size
            
        if wndb == "True":
            wandb.log({"GPT-2 SS IIA Val": matches_val / total_val_samples_processed, "GPT-2 SS IIA Val Loss": total_val_loss / total_val_samples_processed})
        print(f"Validation Accuracy: {matches_val / total_val_samples_processed:.4f}, Validation Loss: {total_val_loss / total_val_samples_processed:.4f}")

def test(model_path, training_model, model, test_data, loss_fn, attribute, token_length_allowed, batch_size, temperature_end, DEVICE, wndb):
    training_model.load_state_dict(torch.load(model_path))
    training_model.eval()

    total_test_samples_processed = 0
    total_test_loss = 0.0

    with torch.no_grad():
        matches_test = 0
        total_test_samples_processed = 0
        total_test_loss = 0
        
        correct_test = {i:[] for i in range(0,12)}
        for sample_no in range(np.array(test_data).shape[0]):
            samples = test_data[sample_no]
            assert np.array(samples).shape == (batch_size, 2, 2)
            # Data Processing
            proceed, base_ids, source_ids, base_label_ids, source_label_ids, source_label, base_label = data_processing(model = model,
                                                                                                            samples = samples, 
                                                                                                            token_length_allowed=token_length_allowed, 
                                                                                                            attribute=attribute,
                                                                                                            DEVICE=DEVICE,
                                                                                                            batch_size=batch_size)
            
            if not proceed:
                continue
            
            temperature = temperature_end
                
            intervened_base_output, predicted_text = training_model(source_ids, base_ids, temperature)
            
            ground_truth_token_id = source_label_ids
            vocab_size = model.tokenizer.vocab_size
            ground_truth_one_hot = F.one_hot(ground_truth_token_id["input_ids"], num_classes=vocab_size)
            ground_truth_one_hot = ground_truth_one_hot.to(dtype=torch.long)
            last_token_output = intervened_base_output[:,-1,:]
            assert ground_truth_one_hot.squeeze(1).shape == last_token_output.shape
            ground_truth_indices = torch.argmax(ground_truth_one_hot.squeeze(1), dim=1)
            ground_truth_indices = ground_truth_indices.to(dtype=torch.long)
            loss = loss_fn(last_token_output, ground_truth_indices)
            total_test_loss += loss.item()
            
            # Calculate accuracy
            predicted_text = [word.split()[0] for word in predicted_text]
            source_label = [word.split()[0] for word in source_label]
            matches_arr = [i for i in range(len(predicted_text)) if predicted_text[i] == source_label[i]]
            matches_test+=len(matches_arr)
            total_test_samples_processed +=batch_size
        
        if wndb == True:
            wandb.log({"GPT-2 SS IIA Test Acc": matches_test / total_test_samples_processed, "GPT-2 SS IIA Test Loss": total_test_loss / total_test_samples_processed})


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--device", default = "cuda:1", help='Device to run the model on')
    parser.add_argument("-a", "--attribute", required = True, help = "name of the attribute on which evaluation is being performned")
    parser.add_argument("-tla", "--token_length_allowed", required=True, type = int, help = "insert the length you would allow the model to train mask")
    parser.add_argument("-method", "--method", required=True, help="to let know if you want neuron masking, das masking or SAE masking")
    parser.add_argument("-e", "--epochs", default=1, type = int, help="# of epochs on which mask is to be trained")
    parser.add_argument("-ef", "--expansion_factor", default=1, help="expansion factor for SAE")
    parser.add_argument("-lr", "--learning_rate", default=0.001, type = int, help="learning rate for the optimizer")
    parser.add_argument("-t", "--task", required=True, help="task to perform, i.e. train or test or total_iia_train")
    parser.add_argument("-svd", "--saved_model_path", default="gpt2/models/saved_model.pth", help="path to the saved model")
    parser.add_argument("-n", "--notes", default="", help = "Any notes you want to write for the wandb graph")
    parser.add_argument("-idd", "--intervention_divided_data", help = "The data which is divided for intervention")
    parser.add_argument("-bs", "--batch_size", default=32, type = int, help="Batch size for training")
    parser.add_argument("-lid", "--layer_intervened", default=0, type = int, help="Layer intervened for the SAE masking")
    parser.add_argument("-wb", "--wndb", default=False, help="Whether to log the data to wandb or not")

    args = parser.parse_args()
    if args.wndb == "True":
        wandb.init(project="sae_concept_eraser")
        wandb.run.name = f"{args.method}-{args.intervention_divided_data}_intervened-e{args.epochs}-b{args.batch_size}-{args.notes}"
    DEVICE = args.device
    layer_intervened = args.layer_intervened

    model, intervened_token_idx, = config(learning_rate = args.learning_rate, token_length = args.token_length_allowed)
    # model.to(DEVICE)
    training_model = my_model(model = model, DEVICE=DEVICE, method=args.method, token_length_allowed=args.token_length_allowed, expansion_factor=args.expansion_factor,
                            layer_intervened=layer_intervened, intervened_token_idx=intervened_token_idx, batch_size=args.batch_size)

    training_model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    
    for name, param in training_model.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}')
    optimizer = optim.Adam(training_model.parameters(), lr=args.learning_rate)

    country_data, continent_data, train_data, val_data, test_data = train_data_processing(args.task, args.intervention_divided_data, args.batch_size)

    #Inserting the temperature
    total_step = 0
    # target_total_step = len(batches) * args.epochs
    #TODO: The total number of batches is total_no_samples/batch_len
    batch_size = args.batch_size
    target_total_step = len(train_data) * args.epochs
    temperature_start = 20.0
    temperature_end = 0.1
    temperature_schedule = (
        t.linspace(t.tensor(temperature_start), t.tensor(temperature_end), int(target_total_step))
        .to(t.bfloat16)
        .to(DEVICE)
    )
    
    temp_idx = 0
    
    with torch.autograd.set_detect_anomaly(True):
        if args.task == "total_iia_train":
            '''
            This correponds to the fact when we are training the model with total intervention and not partial, either on continent or country.
            '''
            train(continent_data = continent_data, country_data = country_data, training_model = training_model, model = model, train_data = train_data, optimizer = optimizer, loss_fn = loss_fn, epochs = args.epochs, token_length_allowed = args.token_length_allowed, attribute = args.attribute, temperature_schedule = temperature_schedule, batch_size = batch_size, DEVICE = DEVICE, wndb = args.wndb)
        
        elif args.task == "train":
        
            train(continent_data = continent_data, country_data = country_data, training_model = training_model, model = model, train_data = train_data, optimizer = optimizer, loss_fn = loss_fn, epochs = args.epochs, token_length_allowed = args.token_length_allowed, attribute = args.attribute, temperature_schedule = temperature_schedule, batch_size = batch_size, DEVICE = DEVICE, wndb = args.wndb)
            # Assuming training_model.l4_mask is your tensor
            l4_mask_cpu = training_model.l4_mask.to('cpu')  # Move tensor to CPU

            # Create a boolean mask where the condition is true
            mask_greater_than_0_5 = l4_mask_cpu > 0
            mask_equal_to_0 = l4_mask_cpu == 0

            # Sum the mask to get the number of elements satisfying the conditions
            num_elements_greater_than_0_5 = mask_greater_than_0_5.sum().item()
            num_elements_equal_to_0 = mask_equal_to_0.sum().item()

            print(f"Number of elements in l4_mask greater than 0.5: {num_elements_greater_than_0_5}")
            print(f"Number of elements in l4_mask equal to 0: {num_elements_equal_to_0}")
            try:
                with open("masking_stats.json","r") as f:
                    data = json.load(f)

                data[f"[{GPT2}-{args.attirbute}] Number of elements in l4_mask > 0.5"] = num_elements_greater_than_0_5
                data[f"[{GPT-2}-{args.attirbute}] num elements in l4 masks = 0"] = num_elements_equal_to_0

                with open("masking_stats.json","w") as f:
                    json.dump(data,f)

            except:

                data[f"[{GPT2}-{args.attirbute}] Number of elements in l4_mask > 0.5"] = num_elements_greater_than_0_5
                data[f"[{GPT-2}-{args.attirbute}] num elements in l4 masks = 0"] = num_elements_equal_to_0

                with open("masking_stats.json","w") as f:
                    json.dump(data,f)


            # Save the model
            torch.save(training_model.state_dict(), f"models/saved_model_{args.intervention_divided_data}_{args.method}_{args.attribute}_{args.model}_{args.epochs}.pth")
            
        elif args.task == "test":
            model_path = args.saved_model_path
            test(model_path, training_model, model,test_data, loss_fn, args.attribute, args.token_length_allowed, batch_size, temperature_end, DEVICE, wndb=args.wndb)

