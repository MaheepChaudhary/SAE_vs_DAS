from Pythia.config import epochs, lr
from imports import *
from model import *
from dataprocessing import *
from visualization import *

input = nn.init.orthogonal_(torch.empty(5,5))
output = torch.eye(5)

# print(input)

# model = Model()

sum_arr = []
sum_arr_epoch = {}


# Create a CSV file and write the header
with open(f'stats/log{name}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy'])


def train(epochs = epochs, lr = lr):

    model = Model()
    
    for epoch in range(epochs):
        sum = 0
        optimizer.zero_grad()
        predicted_output = model(input)
        loss = F.mse_loss(predicted_output, output)
        for i in range(5):
            for j in range(5):
                sum+=int(torch.dot(model.ortho_weight[i], 
                    model.ortho_weight[j]).item())
        # print(sum)
        sum_arr_epoch[epoch] = sum
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch} Loss: {loss.item()}')
        print('-------------------')

        print(model.ortho_weight)
    print('Input:', input)
    print('Output:', output)




def mnist_train(train_loader,
                test_loader,
                train_losses,
                train_counter,
                test_losses,
                test_counter,
                accuracies,
                mnist_model,
                optimizer,
                epochs = epochs, 
                lr = lr):
    
    epoch_loss = []
    epoch_accuracy = []

    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            mnist_model.to(device)    
            optimizer.zero_grad()
            predicted_output = mnist_model(data)
            output = predicted_output.argmax(dim=1)
            correct = torch.sum((output == target).float())
            accuracy = correct.clone().detach().cpu()/len(target)
            loss = F.nll_loss(predicted_output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    # epoch, batch_idx * len(data), len(train_loader.dataset),
                    # 100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.clone().detach().cpu().item())
                accuracies.append(accuracy)
                train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        epoch_loss.append(sum(train_losses)/len(train_losses))
        epoch_accuracy.append((sum(accuracies)/len(accuracies)).item())
        print(f'Epoch: {epoch} Loss: {epoch_loss[-1]}, Accuracy: {epoch_accuracy[-1]}')
        print('-------------------')
        

        with open(f'stats/log{name}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, epoch_loss[-1], str(epoch_accuracy[-1]*100)+"%"])

    return epoch_loss, epoch_accuracy


def dasmat_loss(labels, outputs):
    # Masks for selecting classes 0, 1, and 2 for positive reinforcement
    mask_first_three = (labels == 0) | (labels == 1) | (labels == 2)

    # Masks for selecting classes 3, 4, 5, 6, 7, 8, 9 for negative reinforcement
    mask_remaining = labels > 2

    # Select outputs and labels for the first three 'positive' classes
    outputs_first_three = outputs[mask_first_three]
    labels_first_three = labels[mask_first_three]

    # Select outputs and labels for the 'negative' classes
    outputs_remaining = outputs[mask_remaining]
    labels_remaining = labels[mask_remaining]

    # Compute the usual loss for the first three classes
    loss_first_three = F.nll_loss(outputs_first_three, labels_first_three) if len(labels_first_three) > 0 else torch.tensor(0., requires_grad=True)
    
    # Normalize the loss if needed
    if len(outputs_first_three) > 0:
        loss_first_three = loss_first_three / len(outputs_first_three)

    # Compute the usual loss for the remaining classes
    if len(labels_remaining) > 0:
        loss_remaining = F.nll_loss(outputs_remaining, labels_remaining)
        # Normalize the loss if needed
        loss_remaining = loss_remaining / len(outputs_remaining)
    else:
        loss_remaining = torch.tensor(0., requires_grad=True)

    # Combine the losses
    # Subtracting normalized loss of the remaining classes to penalize accuracy on them
    total_loss = loss_first_three - loss_remaining

    return total_loss


def dasmat_train(train_loader,
                test_loader,
                train_losses,
                train_counter,
                test_losses,
                test_counter,
                accuracies,
                model,
                optimizer,
                epochs = epochs, 
                lr = lr):
    
    epoch_loss = 0
    epoch_accuracy = 0

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):

            #loading into device
            data = data.to(device)
            target = target.to(device)
            model.to(device)

            # forward pass
            optimizer.zero_grad()
            predicted_output = model(data)
            output = predicted_output.argmax(dim=1)
            correct = torch.sum((output == target).float())
            accuracy = correct.clone().detach().cpu()/len(target)
            # loss = F.nll_loss(predicted_output, target)
            loss = dasmat_loss(target, predicted_output)

            # backward pass
            loss.backward(retain_graph=True)
            optimizer.step()
            # print("Gradient of transpose_matrix:", model.transpose_matrix.grad)

            # stats printing 
            if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.clone().detach().cpu().item())
                accuracies.append(accuracy)
                train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            
            # stats storage
            epoch_loss = sum(train_losses)/len(train_losses)
            epoch_accuracy = sum(accuracies)/len(accuracies)
        # print(model.rotate_layer.weight.data.size())
        # print(f"The transpose matrix is {model.transpose_matrix}")
        print(f'Epoch: {epoch} Loss: {epoch_loss}, Accuracy: {epoch_accuracy*100}')
        # print('-------------------')
        
        print(model.binary_mask1)
        print(model.binary_mask2)

        with open(f'stats/log{name}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, epoch_loss, str(epoch_accuracy.item()*100)+"%"])

    return train_losses, train_counter, accuracies


def sparse_model(train_loader,
                test_loader,
                train_losses,
                train_counter,
                test_losses,
                test_counter,
                accuracies,
                model,
                optimizer,
                train,
                epochs = epochs, 
                lr = lr):
    
    epoch_loss = 0
    epoch_accuracy = 0

    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):

            #loading into device
            data = data.to(device)
            target = target.to(device)
            model.to(device)

            # forward pass
            optimizer.zero_grad()
            l1_loss, l2_loss, loss_aggregate, acts, predicted_output = model(data)
            output = predicted_output.argmax(dim=1)
            correct = torch.sum((output == target).float())
            accuracy = correct.clone().detach().cpu()/len(target)

            if train == "sae":
                pred_loss = F.nll_loss(predicted_output, target)
                
                # mutiplying with 50 as it is 50 times less than the loss aggregate
                total_loss = 50*pred_loss + loss_aggregate
                total_loss.backward(retain_graph=True)
            
            elif train == "masksae":
                prediction_loss = dasmat_loss(target, predicted_output)
                total_loss = prediction_loss
                total_loss.backward(retain_graph=True)

            optimizer.step()
            if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(total_loss.clone().detach().cpu().item())
                accuracies.append(accuracy)
                train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        
            
            # stats storage
            epoch_loss = sum(train_losses)/len(train_losses)
            epoch_accuracy = sum(accuracies)/len(accuracies)

        print(f"reconstruction loss is {l2_loss}")
        print(f'Epoch: {epoch} Loss: {epoch_loss}, Accuracy: {epoch_accuracy*100}')
        # print('-------------------')
        # print(model.binary_mask.grad)

        with open(f'stats/log{name}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, epoch_loss, str(epoch_accuracy.item()*100)+"%"])
        
    return train_losses, test_losses, accuracies
            

def count_classes(dataset):
    labels = [data[1] for data in dataset]
    label_counts = Counter(labels)
    return label_counts

def eval(model, train):

    model.eval()  # Set the model to evaluation mode

    # Initialize counters
    correct_per_class = torch.zeros(10)  # Assuming there are 10 classes
    total_per_class = torch.zeros(10)
    accuracy_per_class = [0]*10
    # print(accuracy_per_class)

    with torch.no_grad():  # No need to track gradients for evaluation
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            if train == "sae":
                l1_loss, l2_loss, loss_aggregate, acts, predicted_output = model(data)
            else:
                predicted_output = model(data)
            # pred = outputs.data.max(1, keepdim=True)[1]
            pred = predicted_output.argmax(dim=1)
            # correct += pred.eq(target.data.view_as(pred)).sum()
            accuracy_per_class[target.item()]+=pred.eq(target.data.view_as(pred)).sum()
    accuracy_per_class = [i.item() for i in accuracy_per_class]
    counted_classes = count_classes(test_loader.dataset)
    normalized_accuracy_per_class = [accuracy_per_class[i]/counted_classes[i] for i in range(10)]
    print(accuracy_per_class)
    print(normalized_accuracy_per_class)
    return normalized_accuracy_per_class




parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('-t', '--train', type=str, default='true')
parser.add_argument("-a", "--description", type=str)
parser.add_argument("-m", "--model_path", type=str, default='mnist_models/2_0133.pt')
args = parser.parse_args()



if __name__ == '__main__':

    train_loader, test_loader = data_loader()
    train_losses = []
    train_counter = []
    test_losses = []
    accuracies = [] 
    test_counter = [i*len(train_loader.dataset) for i in range(args.epochs + 1)]


    if args.train == 'vanilla': train(args.epochs, args.lr)

    elif args.train == 'mnist':

        model = mnistmodel()
        print(model)
        model.train()
        # model = MNISTModel()
        optimizer = optim.SGD(model.parameters(), 
                              lr=lr, 
                              momentum=momentum)

        
        train_losses_, acc = mnist_train(train_loader,
                                        test_loader,
                                        train_losses,
                                        train_counter,
                                        test_losses,
                                        test_counter,
                                        accuracies,
                                        model, 
                                        optimizer, 
                                        args.epochs, 
                                        args.lr
                                        )

        torch.save(model.state_dict(), f'mnist_models/{name}.pt')
        visualization(train_losses_, acc, args.description)

        # model.load_state_dict(torch.load("mnist_models/model.pth"))

        accuracy = eval(model)
        visualization_each_class(accuracy, args.description)

    elif args.train == 'dasmat':

            model = dasmat(args.model_path)   
            model = model.to(device) 
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            train_losses_, test_losses_, acc = dasmat_train(train_loader,
                                                            test_loader,
                                                            train_losses,
                                                            train_counter,
                                                            test_losses,
                                                            test_counter,
                                                            accuracies,
                                                            model,
                                                            optimizer,
                                                            args.epochs, 
                                                            args.lr
                                                            )

            torch.save(model.state_dict(), f'dasmat_models/{name}.pt')
            visualization(train_losses_, acc, args.description)


            accuracy = eval(model)
            visualization_each_class(accuracy, args.description)

    elif args.train == 'sae':
            
            ae_cfg = AutoEncoderConfig(n_instances = batch_size_train,
                            l1_coeff = 0.00025,
                            tied_weights = True
                            )
            
            model = sparse_encoder(args.model_path, ae_cfg)
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            train_losses_, test_losses_, acc = sparse_model(train_loader,
                                                            test_loader,
                                                            train_losses,
                                                            train_counter,
                                                            test_losses,
                                                            test_counter,
                                                            accuracies,
                                                            model, 
                                                            optimizer, 
                                                            args.train,
                                                            args.epochs, 
                                                            args.lr
                                                            )
    
            torch.save(model.state_dict(), f'sae_models/{name}.pt')
            visualization(train_losses_, acc, args.description)
    
            accuracy = eval(model, args.train)
            visualization_each_class(accuracy, args.description)

    elif args.train == 'masksae':
            
            ae_cfg = AutoEncoderConfig(n_instances = batch_size_train,
                            l1_coeff = 0.00025,
                            tied_weights = True
                            )
            
            model_path = "sae_models/6_2107.pt"
            model = mask_sparse_encoder(model_path, args.model_path, ae_cfg)
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            train_losses_, test_losses_, acc = sparse_model(train_loader,
                                                            test_loader,
                                                            train_losses,
                                                            train_counter,
                                                            test_losses,
                                                            test_counter,
                                                            accuracies,
                                                            model, 
                                                            optimizer, 
                                                            args.train,
                                                            args.epochs, 
                                                            args.lr
                                                            )
    
            torch.save(model.state_dict(), f'masksae_models/{name}.pt')
            visualization(train_losses_, acc, args.description)
    
            accuracy = eval(model, args.train)
            visualization_each_class(accuracy, args.description)
    

    else:
        model = torch.load(f'dasmat_models/Tue_Apr_30_02:11:32_2024.pt')
        print(model)