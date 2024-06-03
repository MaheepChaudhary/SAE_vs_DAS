from imports import *
from config import *
from model import *
from dataprocessing import *


def train(DEVICE, 
        epochs, 
        lr,
        mini_batch,
        evaluation,
        batch_size_train,
        activation_dim,
        residual_layer,  
        dict_embed_path,
        method,
        attn_dict_path,
        mlp_dict_path,
        resid_dict_path,
        expansion_factor):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    wandb.init(project="sae_concept_eraser")
    wandb.run.name = f"[{evaluation}]-{residual_layer}-{method}_b{batch_size_train}_e{epochs}"

    print("passed dict emned path", dict_embed_path)

    new_model = my_model(DEVICE = DEVICE,
                        dict_embed_path = dict_embed_path,
                        attn_dict_path = attn_dict_path,
                        mlp_dict_path = mlp_dict_path,
                        resid_dict_path = resid_dict_path,
                        resid_layers=residual_layer,
                        method = method,
                        activation_dim = activation_dim,
                        expansion_factor=expansion_factor,
                        epochs = epochs).to(DEVICE)
    

    optimizer = t.optim.Adam(new_model.parameters(), lr = lr)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    batches = get_data(DEVICE, train = True, ambiguous = False, batch_size=batch_size_train) # by default the ambigous is True
    
    # print(len(batches))
    
    random.shuffle(batches)
    # if mini_batch == 0:
    #     train_batches = batches[0:1000]
    # elif mini_batch == 1:
    # train_batches = batches

    total_step = 0
    target_total_step = len(batches) * epochs
    temperature_start = 50.0
    temperature_end = 0.1
    temperature_schedule = (
        t.linspace(temperature_start, temperature_end, target_total_step)
        .to(t.bfloat16)
        .to(DEVICE)
    )
    

    label_idx = 0
    # print(len(batches))
    temp_idx = 0
    for epoch in range(epochs):
        losses = []
        len_batches = len(batches)
        for i in tqdm(range(len_batches)):

            text = batches[i][0]
            
            if evaluation == "profession":
                labels = batches[i][1] # true label, if [2] then spurious label. We will be training the model in hope that mask will learn which concepts to mask. 
            elif evaluation == "gender":
                labels = batches[i][2]
            
            temprature = temperature_schedule[temp_idx]


            logits = new_model(text, temperature=temprature)
            # print(logits.shape)
            # print(labels.float)
            loss = criterion(logits, labels.float())
            optimizer.zero_grad()
            try:
                loss.backward()
            except:
                pass
            optimizer.step()
            losses.append(loss.item())
            temp_idx += 1
            # if len(losses) % 10 == 0:
                # print(f"Epoch: {epoch}, Loss: {np.mean(losses)}")
            
            if DEVICE == "cuda:1":
                t.cuda.empty_cache()
            
            elif DEVICE == "mps":
                t.mps.empty_cache()
        
        wandb.log({"Full Data Gender de-baising Losses": np.mean(losses)})
                    
                
    t.save(new_model.state_dict(), f"saved_models/{evaluation}-{residual_layer}-{method}_b{batch_size_train}_e{epochs}.pth")
                
def eval(DEVICE, 
        saved_model_path,
        epochs,
        lr,
        mini_batch,
        evaluation,
        batch_size_train,
        activation_dim,
        residual_layer,
        method,
        dict_embed_path,
        attn_dict_path,
        mlp_dict_path,
        resid_dict_path,
        expansion_factor):

    wandb.init(project="sae_concept_eraser")
    wandb.run.name = f"{evaluation}-{saved_model_path}-b1"
    
    new_model = my_model(DEVICE = DEVICE,
                    dict_embed_path = dict_embed_path,
                    attn_dict_path = attn_dict_path,
                    mlp_dict_path = mlp_dict_path,
                    resid_dict_path = resid_dict_path,
                    resid_layers=residual_layer,
                    method = method,
                    activation_dim = activation_dim,
                    expansion_factor=expansion_factor,
                    epochs = epochs).to(DEVICE)

    # Load the state dictionary

    new_model.load_state_dict(t.load(saved_model_path))
    new_model = new_model.to(DEVICE)
    new_model.eval()


    batches = get_data(DEVICE, train = False, ambiguous=False) # by default the ambigous is True
    label_idx = 0
    len_batches = len(batches)
    corrects = []
    total = 0
            
    # subgroups = get_subgroups(train=False, ambiguous=False)
    # for label_profile, batches in subgroups.items():
    # for i in tqdm(range(len_batches)):
        
    #     text = batches[i][0]
    #     labels = batches[i][label_idx+1] 
        
    with t.no_grad():

        len_batches = len(batches)
        for i in tqdm(range(len_batches)):
            text = batches[i][0]
            
            if evaluation == "profession":
                labels = batches[i][1] # true label, if [2] then spurious label. We will be training the model in hope that mask will learn which concepts to mask. 
            elif evaluation == "gender":
                labels = batches[i][2]
            
            # acts = get_acts(text)
            logits = new_model(text, temperature=0.1)
            # preds = (logits > 0.0).long()
            preds = (logits > 0.0).long()
            corrects.append((preds == labels).float())
        
        accuracy = t.cat(corrects).mean().item()

    wandb.log({"Full Data Accuracy": accuracy})
    

def eval_on_subgroups(DEVICE, 
        saved_model_path,
        epochs,
        lr,
        mini_batch,
        evaluation,
        batch_size_train,
        activation_dim,
        residual_layer,
        method,
        dict_embed_path,
        attn_dict_path,
        mlp_dict_path,
        resid_dict_path,
        expansion_factor):
    
    # Here we will find the accuracy of the subgroups for both the vanilla model and the model with the mask.

    wandb.init(project="sae_concept_eraser")

    new_model = my_model(DEVICE = DEVICE,
                        dict_embed_path = dict_embed_path,
                        attn_dict_path = attn_dict_path,
                        mlp_dict_path = mlp_dict_path,
                        resid_dict_path = resid_dict_path,
                        resid_layers=residual_layer,
                        method = method,
                        activation_dim = activation_dim,
                        expansion_factor=expansion_factor,
                        epochs = epochs).to(DEVICE)

    # Load the state dictionary
    new_model.load_state_dict(t.load(saved_model_path))
    new_model = new_model.to(DEVICE)
    new_model.eval()

    subgroups = get_subgroups(DEVICE, train=False, ambiguous=False)

    with t.no_grad():
        for label_profile, batches in subgroups.items():
            corrects = []
            total = 0
            for i in tqdm(range(len(batches))):
                text = batches[i][0]
                labels = label_profile[0] # true label, if [2] then spurious label. We will be training the model in hope that mask will learn which concepts to mask.
                logits = new_model(text, temperature = 0.1)
                preds = (logits > 0.0).long()
                corrects.append((preds == labels).float())
            
            accuracy = t.cat(corrects).mean().item()
            
            if label_profile == (0, 0):
                wandb.run.name = "Vanilla-acc-Male_Prof.-{saved_model_path}-b1"
                wandb.log({"Groups Accuracy": accuracy})
                print(f"Accuracy for Male Professor is:", accuracy)
            elif label_profile == (0, 1):
                wandb.run.name = "Vanilla-acc-Female_Prof.-{saved_model_path}-b1"
                wandb.log({"Groups Accuracy": accuracy})
                print(f"Accuracy for Female Professor is:", accuracy)
            elif label_profile == (1, 0):
                wandb.run.name = "vanilla-acc-Male_Nurse-{saved_model_path}-b1"
                wandb.log({"Groups Accuracy": accuracy})
                print(f"Accuracy for Male Nurse is:", accuracy)
            elif label_profile == (1, 1):
                wandb.run.name = "Vanilla-acc-Female_Nurse-{saved_model_path}-b1"
                wandb.log({"Groups Accuracy": accuracy})
                print(f"Accuracy for Female Nurse is:", accuracy)

        # print(f'Accuracy for {label_profile}:', test_probe(oracle, get_acts, batches=batches, label_idx=0))

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('-e', '--epochs', default=50, type=int, help='number of epochs')
    argparser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate')
    argparser.add_argument('-btr', '--batch_size_train', type=int, required=True, help='batch size for training')
    argparser.add_argument('-d', '--device', type=str, required=True, help='device to be used')
    argparser.add_argument('-layer', '--residual_layer', type=int, required=True, help='residual layer to be used intervened in the model w/ range 0->4')
    argparser.add_argument('-ad', '--activation_dim', default=512, type=int, help="activation dimension")
    argparser.add_argument('-ef', '--expansion_factor', default=64, type=int, help="expansion factor")
    
    argparser.add_argument("-dpath", "--dict_embed_path", 
                           default="/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/embed",
                           type=str, 
                           help="dictionary embedding path")
    
    argparser.add_argument("-atpath", "--attn_dict_path", 
                           default="/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/attn_out_layer",
                           type=str, help="attention dictionary path")
    
    argparser.add_argument("-mpath", "--mlp_dict_path", 
                           default="/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/mlp_out_layer",
                           type=str, help="mlp dictionary path")
    
    argparser.add_argument("-rpath", "--resid_dict_path", 
                           default="/Users/maheepchaudhary/pytorch/Projects/concept_eraser_research/DAS_MAT/baulab.us/u/smarks/autoencoders/pythia-70m-deduped/resid_out_layer",
                           type=str, help="residual dictionary path")
    
    argparser.add_argument("-mb", "--mini_batch", required=True, default = 0, help="0 if you want mini batch and 1 if you want full batch")
    argparser.add_argument("-eval", "--evaluation", required=True, type=str, help="evaluation metric, either profession or gender")
   # argparser.add_argument("-pp", "--probe_path", required=True, type=str, help="path of probe model to be used")
    argparser.add_argument("-svd","--saved_model_path", default = "new_model.pth", type=str, help="path to save the model")
    
    argparser.add_argument("-task", "--task", required=True, type=str, help="task to be performed, i.e. train, eval or eval_on_subgroups")
    argparser.add_argument("-nds", "--method", required=True, type=str, help="method to be used, i.e. neuron masking, sae masking or das masking")
    
    
    args = argparser.parse_args()
    # args.residual_layer = [int(i) for i in args.residual_layer]
    
    
    # with open("probe_shift.pkl", "rb") as f:
    # with open(args.probe_path, "rb") as f:
    #     probe = pkl.load(f)

    
    if args.task == "train":
        
    
        train(DEVICE=args.device,
            epochs=args.epochs,
            lr = args.learning_rate,
            mini_batch=args.mini_batch,
            evaluation=args.evaluation,
            batch_size_train=args.batch_size_train,
            activation_dim=args.activation_dim,
            residual_layer=args.residual_layer,
            method=args.method,
            dict_embed_path=args.dict_embed_path,
            attn_dict_path=args.attn_dict_path,
            mlp_dict_path=args.mlp_dict_path,
            resid_dict_path=args.resid_dict_path,
            expansion_factor=args.expansion_factor)
        
    
    elif args.task == "eval":
        eval(args.device, 
            args.saved_model_path,
            epochs=args.epochs,
            lr = args.learning_rate,
            mini_batch=args.mini_batch,
            evaluation=args.evaluation,
            batch_size_train=args.batch_size_train,
            activation_dim=args.activation_dim,
            residual_layer=args.residual_layer,
            method=args.method,
            dict_embed_path=args.dict_embed_path,
            attn_dict_path=args.attn_dict_path,
            mlp_dict_path=args.mlp_dict_path,
            resid_dict_path=args.resid_dict_path,
            expansion_factor=args.expansion_factor)
    
    elif args.task == "eval_on_subgroups":
        eval_on_subgroups(args.device, 
            args.saved_model_path,
            epochs=args.epochs,
            lr = args.learning_rate,
            mini_batch=args.mini_batch,
            evaluation=args.evaluation,
            batch_size_train=args.batch_size_train,
            activation_dim=args.activation_dim,
            residual_layer=args.residual_layer,
            method=args.method,
            dict_embed_path=args.dict_embed_path,
            attn_dict_path=args.attn_dict_path,
            mlp_dict_path=args.mlp_dict_path,
            resid_dict_path=args.resid_dict_path,
            expansion_factor=args.expansion_factor)
    
    




# python main.py -e 10 -btr 16 -d cuda:1 -layer "4" -pp probe_shift.pkl -task train -eval profession -nds "neuron masking" -dpath ./dictionary_learning/dictionaries/pythia-70m-deduped/embed -atpath ./dictionary_learning/dictionaries/pythia-70m-deduped/attn_out_layer -mpath ./dictionary_learning/dictionaries/pythia-70m-deduped/mlp_out_layer -rpath ./dictionary_learning/dictionaries/pythia-70m-deduped/resid_out_layer -mb 1
