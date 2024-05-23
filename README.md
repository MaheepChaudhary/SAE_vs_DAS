The information about the project experimentation with task can be found here: https://api.wandb.ai/links/counterfactuals/5gjwu6xk

The files in the repository contain the following:

1. `sae_eraser.ipynb` contains all the main coding in the project including the code of masks and model. 
2. `sae_eraser copy.ipynb` contains the code for dummy model, to ensure that the model runs (not to be engaged in).
3. `main.py` contains the code for experimentation on the mnist dataset for concept eraser. We tried to erase different digit information in the mnist dataset. 

## Executing the file:

These are the arguments that needs to be passed while running the `main.py` file. 

```
argparser.add_argument('-dict','download_dictionary', action='store_true', help="a boolean, helping to get you dictionary if you don't have it.")
argparser.add_argument("-dr", 'data_dir', type=str, help='directory to save the data')
argparser.add_argument('-e','--epochs', default=15, type=int, help='number of epochs')
argparser.add_argument('-lr','--lr', default=0.001, type=float, help='learning rate')
argparser.add_argument('-btr','batch_size_train', type=int, help='batch size for training')
argparser.add_argument('-bts','batch_size_test', type=int, help='batch size for testing')
argparser.add_argument("-d",'device', type=str, help='device to be used')
argparser.add_argument("-layer",'residual layer', type=list, help="residual layer to be used interevened in the model")
```

```
pip install requirements.txt
python main.py -dict -dr "path" -btr 1 -bts 1 -d "cuda:0" -layer [3,15]
```

## File Information:

`imports.py`: It contains the list of imported modules

`config.py`: It contains the information about different hyperparameters.

`main.py`: It contain the execution taking support from other `.py` files. 