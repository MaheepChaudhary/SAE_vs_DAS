## Steps to prepare ravel data:

- Firstly make the dataset for preparing simple file: "country_data.json"
- After that a file is prepared consisting of cities in which the model is comforatable in predicting the cities, giving around 100% accuracy:

  - For each string check if it is giving is given correct by the model. If yes, then record it. Furthermore, also analyse the different length of sentences, and see which are the ones that does not break city: _61 is the one that does not break city and gives the city name in format GAlaxendria._
  - Do it for both the city and continent; and record the cities for that are overlapping. - _The total number of overlapping cities are 101 for llama3b. The total number of samples in intervention continent data is 14645 and the country data is 15251_.
  - The total

- After making the gpt2 from this method and also including the same sentences as base and source. Also including the sentences with same label of base and source our samples increase form 1.5k for country and continent to 4680 and 5200. Of course the overlapping cities remains the same: 40 cities.

## GPT-2:

- One of the things that could be noted in this whole experimentation is that one of the attributes, i.e. either "continent" or "country" gets sacrificed when one starts increasing for every technique, with an exception of DAS.
- It could also be the case that each of the techniques has their own appropriate temperature values. We could experiment on it in the future, as we could clearly see the transistion was not smooth for DAS and SAE, meanwhile the temperature value was set based on the optimal performance of neuron masking. I don't know if it should be done? @Atticus

### GPT-2 Neuron Masking Evalutaion.

- It is to be noted that the results are not in favour it seems like the making value are being favoured for the value that is not being intervened.
- How can we explain the mismatch between the accuracy of the country and continent accuracy for the training and when we are individually targetting them.
- One of the things to be noted is that although the accuracy of the country comes to be $1$ when continent is intervened but when we applied sigmoid, it does not happen the same.
- The training accuracy cannot be equal to the 1/2 of the country + continent as they also include the data for testing, validation and even training. Also, i don't think we can do a partiality with the performance of the country or continent as it does not matter if they are intervened or not, as they are initialized to have 0.5 contribution of both the source and base.
- There might be an issue in the transistion to smaller values of the temperature, as a result, let's see if the temperature instead from 20-0.1 to 10-0.1 shows what kind of values.
- The learning rate also does not provide much performance gain when it is just 0.001, but provides significant gain, when it is 0.01.

### GPT-2 SAE Evalutation

- Now i will be evaluating on whole batch and not just per batch on wandb.
