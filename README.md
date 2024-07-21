# Steps to prepare ravel data:

* Firstly make the dataset for preparing simple file: "country_data.json"
* After that a file is prepared consisting of cities in which the model is comforatable in predicting the cities, giving around 100% accuracy:
   * For each string check if it is giving is given correct by the model. If yes, then record it. Furthermore, also analyse the different length of sentences, and see which are the ones that does not break city: *61 is the one that does not break city and gives the city name in format GAlaxendria.* 
   * Do it for both the city and continent; and record the cities for that are overlapping. - *The total number of overlapping cities are 101 for llama3b. The total number of samples in intervention continent data is 14645 and the country data is 15251*. 
   * The total

* After making the gpt2 from this method and also including the same sentences as base and source. Also including the sentences with same label of base and source our samples increase form 1.5k for country and continent to 4680 and 5200. Of course the overlapping cities remains the same: 40 cities. 
