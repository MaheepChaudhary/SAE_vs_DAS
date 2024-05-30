from imports import *

def eval_on_vanilla_gpt(DEVICE, model, model_name, dataset, attribute, tokenizer):
    
    correct = 0
    total = 0
    for i in tqdm(range(len(dataset))):
        data, label = dataset[i]
        with model.trace() as runner:
            with runner.invoke(data) as invoker:
                logits = model.lm_head.output.save()

        probabilities = torch.softmax(logits[:, -1, :], dim=-1)

        # Get the most likely next token ID
        next_token_id = torch.argmax(probabilities, dim=-1).item()

        # Decode the token ID to a string
        next_token = tokenizer.decode(next_token_id)
        
        if next_token == label:
            correct+=1
            total+=1
            # print("Correct Answer: ", label)
            # print("Predicted Answer: ", next_token)
            # print("Correct!")
        
        elif next_token != label:
            total+=1
    
    accuracy = (correct/total)*100
    print(f"The accuracy of the model on  evaluation for {attribute} is {(correct/total)*100}")
    return accuracy