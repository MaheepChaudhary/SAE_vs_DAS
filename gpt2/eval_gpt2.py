from imports import *

def eval_on_vanilla_gpt(DEVICE, model, model_name, dataset, attribute, tokenizer, type_acc):
    
    correct = 0
    total = 0
    correct_arr = []
    for i in tqdm(range(len(dataset))):
        data, label = dataset[i]
        inputs = tokenizer.encode(data, return_tensors='pt').to(DEVICE)
        inputs.to(DEVICE)
        with model.trace(inputs):
            logits = model.lm_head.output.argmax(dim = -1).save()

        # probabilities = torch.softmax(logits[:, -1, :], dim=-1)

        # Get the most likely next token ID
        # next_token_id = torch.argmax(probabilities, dim=-1).item()
        
        # Get the top 5 token IDs
        
        # if i == 100:
        #     break
        
        if type_acc == "top1":
            # next_token_id = torch.argmax(probabilities, dim=-1).item()
            next_token = tokenizer.decode(logits[0][-2], skip_special_tokens=True).strip()
            # print(f"The next token is {next_token}")
            # print(f"The correct token is {label}")
            # print()
                
    
            if next_token == label:
                correct+=1
                total+=1
                correct_arr.append([data, label])
                
            elif next_token != label:
                total+=1

            if i % 100 == 0:
                print(correct)
    
            
        elif type_acc == "top5":
            
            top5_probabilities, top5_token_ids = torch.topk(probabilities, 5, dim=-1)
            # Decode the top 5 token IDs to strings
            top5_tokens = [tokenizer.decode(token_id.item()) for token_id in top5_token_ids[0]]
            # top5_tokens = [str(token).strip() for token in top5_tokens]
            print(f"The top 5 tokens are {top5_tokens}")
            print(f"The correct token is {label}")
            print()
            
            if label in top5_tokens:
                correct+=1
                total+=1
                # print("Correct Answer: ", label)
                # print("Predicted Answer: ", top5_tokens)
            
            elif label not in top5_tokens:
                total+=1
                # print(f"Correct Answer: {label}")
                # print(f"Predicted Answer: {top5_tokens}")
                # print("Incorrect!")
                # print()
            
            print(correct)
        
        elif type_acc == "top10":
            
            top10_probabilities, top10_token_ids = torch.topk(probabilities, 10, dim=-1)
            # Decode the top 10 token IDs to strings
            top10_tokens = [tokenizer.decode(token_id.item()) for token_id in top10_token_ids[0]]
            top10_tokens = [str(token).strip() for token in top10_tokens]
            
            if label in top10_tokens:
                correct+=1
                total+=1
            
            elif label not in top10_tokens:
                total+=1
            
            print(correct)


        # Decode the token ID to a string
        # next_token = tokenizer.decode(next_token_id)

        
        # if next_token == label:
        #     correct+=1
        #     total+=1
        #     print("Correct Answer: ", label)
        #     print("Predicted Answer: ", next_token)
        #     print("Correct!")
        
        # elif next_token != label:
        #     print("Correct Answer: ", label)
        #     print("Predicted Answer: ", next_token)
        #     total+=1

        
    
    accuracy = (correct/total)*100
    print(f"The accuracy of the model on  evaluation for {attribute} is {(correct/total)*100}")
    return accuracy, correct_arr
