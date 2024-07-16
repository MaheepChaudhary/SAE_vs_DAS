# 👀🐧 Problem Statement


# 🧑‍🔬🧪 Experimentations

## Experiment 1: Scrutinizing the orthogonal matrix


* **(Q1)** *Therefore the question arises, how is it different than initiating a matrix with a gaussian distribution?*
* **(Q2)** *Is it just a learnable matrix or does initializing it with orthogonal vectors produces some change?*


## Experiment 2: Training DAS orthogonal matrix to remove preserve a task-specific concept. 

### Experiment 2.1 MNIST dataset -> treating digit as tasks. 

Initially, we train the MNIST model with `2` linear layers for all the classes of the dataset with architecture

```
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=320, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
```


with training stats as 

![alt text](mnist/stats/fig_Mon_Apr_29_01:30:03_2024.png "Mon_Apr_29_01:30:03_2024.png")

For more stats one can refer to the stats folder, based on the image name. 

The concept of DAS uses the trainable orthogonal matrix to project the hidden representation into a sub-space and intervenes in that space using counterfactual embeddings and re-originate the original embedding. 

$$
x_{n+1} = W_{orth}^{-1}(\psi(W_{orth}(f_{layer}(x_{n}))))
$$

Now the question arises how would be intervene using the trained model to do intervention and would it even be beneficial. According to me, that can only work to remove the concepts 

Either we can use gradient ascent or KL divergence to accomplish the same. 

We can also use interchange intervention, meanwhile updating the weight of orthogonal matrix in the frozen model, by taking them as output in each batch iteration from the learnable model. 

---

### Experiment 2.1.1

MNIST Model architecture:

```
mnistmodel(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=320, out_features=160, bias=True)
  (fc2): Linear(in_features=160, out_features=80, bias=True)
  (fc3): Linear(in_features=80, out_features=10, bias=True)
)
```

The accuracy and loss curve for the trained mnist model:

![alt text](mnist/stats/fig_2_0133.png)

The accuracy of test dataset for individual class is

![alt text](mnist/stats/fig_2_0133_each_class.png)

<!-- KL Divergence and activation attribution might be used for loss -->

We experiment using gradient descent and ascent method by taking loss function to subtract the loss of first 3 classes prediction from last 7 classes prediction. We aimed to erase the ability of the model to recognize any digit other than digits `0`, `1` and `2`; and remove all the digits other than this from `3-9` in mnist, after the hidden representation is projected using DAS.

We trained the new model, treating the old model as the support model and inserting the orthogonal matrix as learnable matrix.

#### We did this using three methods:

**One Layer Intervention**

* Using the just orthogonal matrix.

    For this we hope that nothing will change. It changes the value to make the    remaining classes `0`, even when the mask is not implemented and just the orthogonal matrix is computed. Although the loss is implemented which supports the loss of learning of first 3 classes, i.e. `0`, `1` and `2`.

    The test accuracy after implementing looks like:

    ![alt](https://github.com/MaheepChaudhary/DAS_MAT/blob/mnist/main/stats/fig_2_0206_each_class.png)

* Using just mask matrix.

    With the binary, the accuracy of the model remains the same in training and when looked at the mask. It mostly remains the same. It comes to me as a surprise that it is able to push the classes down. 
    
    ![alt](https://github.com/MaheepChaudhary/DAS_MAT/blob/mnist/main/stats/fig_3_1511_each_class.png)

    I think one of its reasons is its challenging rate of learning. Therefore, we train it using the batch size $1$ training loop with less learning rate and less momentum. However, it also did not give much changed results. 

* Using orthogonal + mask.
    
    Now we will implement the orthogonal rotation with mask. My guess is that orthogonal matrix will act as an upper bound and will not the take the contribution of the mask and the performance will be same as originally. 

    ![alt](https://github.com/MaheepChaudhary/DAS_MAT/blob/mnist/main/stats/fig_3_1549_each_class.png)

* Using orthogonal + mask + inversion. 

    If we are rotating, the it should again signify the contribution of the mask we are using. Hence, it should produce the graph equivalent to just "mask matrix" graph. 
    
    *I GOT WRONG HERE. Don't know why this performance?* @DrGeiger could you please explain?

    ![alt](https://github.com/MaheepChaudhary/DAS_MAT/blob/mnist/main/stats/fig_3_1552_each_class.png)



**Two Layer Intervention**

* 2 rotation layers

    **(Q1)** *One thing that i am not sure is the vector we rotate is just passed from the relu activation function. The relu erases the information, so should we keep it or not? As of now I am keeping it.*

    I think it should give perfect result as given by the single rotation layers. 

    ![alt](https://github.com/MaheepChaudhary/DAS_MAT/blob/mnist/main/stats/fig_3_1604_each_class.png)

* 2 rotation layer with mask in first layer. 

    My guess is that orthogonal matrix will act as an upper bound and will not the take the contribution of the mask and the performance will be same as originally. 

    ![alt](https://github.com/MaheepChaudhary/DAS_MAT/blob/mnist/main/stats/fig_3_1607_each_class.png)


* 2 rotation layer with mask and inverse layer in first layer. 

    It performance was far better than first we tried it. I think this is due to the presence of second rotation layer. 

    ![alt](https://github.com/MaheepChaudhary/DAS_MAT/blob/mnist/main/stats/fig_3_1612_each_class.png)
    
* 2 rotation layer with 2 mask and inverse layer in first layer.  

    My guess is that orthogonal matrix will act as an upper bound and will not the take the contribution of the mask and the performance will be same as originally.

    ![alt](https://github.com/MaheepChaudhary/DAS_MAT/blob/mnist/main/stats/fig_3_1622_each_class.png)

* 2 rotation layer with 2 mask and 2 inverse layers. 

    When we do this, one of the classes that we don't want to lose, gets retained. i don't know how. Can you help me interpret this @Dr.Geiger?

    ![alt](https://github.com/MaheepChaudhary/DAS_MAT/blob/mnist/main/stats/fig_3_1624_each_class.png)



# 🎯 Tasks

✅ Implement the orthogonal matrix that remains constant. \
✅ Implement the binary mask also with the orthogonal matrix. \
✅ Use it for each layer. \
✅ Implement a DAS complete architecture and experiment on it.  
✅ See if the gradients flow is changing the masking function, extract the masking function for each class. \
✅ Make the network with 3 linear layers to experiment more efficiently. 
✅ Figure out the overall experimentation of the project. 
✅ Complete the implementation of sparse autoencoders inside mnist. 
✅ Insert a mask inside the sparse autoencoder. 

# 📑 APPENDIX


