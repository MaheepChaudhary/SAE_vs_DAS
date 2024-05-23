from imports import *
from config import *
from model import *
from dataprocessing import *


def train(DEVICE, 
        epochs, 
        lr,
        mini_batch,
        evaluation):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    wandb.init(project="sae_concept_eraser")
    wandb.run.name = "gender_Lastlinear-mask_amb(false)_b1_e15_mini-batch-test1k"
    # wandb.run.name = "gender_Lastlinear-mask_probe_amb(false)_b1_e3"

    new_model = my_model().to(DEVICE)

    optimizer = t.optim.Adam(new_model.parameters(), lr = lr)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    batches = get_data(train = True, ambiguous = False) # by default the ambigous is True
    
    random.shuffle(batches)
    if mini_batch:
        train_batches = batches[0:1000]
    else:
        train_batches = batches

    label_idx = 0
    # print(len(batches))

    for epoch in range(epochs):
        losses = []
        len_batches = len(train_batches)
        for i in tqdm(range(len_batches)):
            text = train_batches[i][0]
            
            if evaluation == "profession":
                labels = train_batches[i][1] # true label, if [2] then spurious label. We will be training the model in hope that mask will learn which concepts to mask. 
            elif evaluation == "gender":
                labels = train_batches[i][2]
                
            # print(labels.float)
            logits = new_model(text)
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
            if len(losses) % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {np.mean(losses)}")
                wandb.log({"Gender de-baising Losses": np.mean(losses)})
                losses = []
                


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('data_dir', type=str, help='directory to save the data')
    argparser.add_argument('-e','--epochs', default=15, type=int, help='number of epochs')
    argparser.add_argument('-lr','--lr', default=0.001, type=float, help='learning rate')
    argparser.add_argument('-btr','batch_size_train', type=int, help='batch size for training')
    argparser.add_argument('-bts','batch_size_test', type=int, help='batch size for testing')
    argparser.add_argument("-d",'device', type=str, help='device to be used')
    argparser.add_argument("-layer",'residual layer', type=str, help="residual layer to be used interevened in the model")
    argparser.add_argument("-activation_dim", type=int, help="activation dimension")
    argparser.add_argument("-ef", "--expansion_factor", default = 64, type=int, help="expansion factor")
    argparser.add_argument("-dict_embed_path", type=str, help="dictionary embedding path")
    argparser.add_argument("-attn_dict_path", type=str, help="attention dictionary path")
    argparser.add_argument("-mlp_dict_path", type=str, help="mlp dictionary path")
    argparser.add_argument("-resid_dict_path", type=str, help="residual dictionary path")
    argparser.add_argument("-mb", "mini_batch", action='store_true', help="for just training on 1000 samples of training data, then yes!")
    argparser.add_argument("-eval", "evaluation", type=str, help="evaluation metric, either profession or gender")
    argparser.add_argument("-pp", "probe_path", type=str, help="path of probe to be used")
    
    
    args = argparser.parse_args()
    
    
    # with open("probe_shift.pkl", "rb") as f:
    with open(args.probe_path, "rb") as f:
        probe = pkl.load(f)
    
    train(args.device, args.epochs, args.lr, args.mini_batch, args.evaluation)





