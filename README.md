The information about the project experimentation with task can be found here: https://api.wandb.ai/links/counterfactuals/5gjwu6xk

The files in the repository contain the following:

1. `sae_eraser.ipynb` contains all the main coding in the project including the code of masks and model. 
2. `sae_eraser copy.ipynb` contains the code for dummy model, to ensure that the model runs (not to be engaged in).
3. `main.py` contains the code for experimentation on the mnist dataset for concept eraser. We tried to erase different digit information in the mnist dataset. 

## Executing the file:

Before running the code, these are the following this to be downloaded:

1. The weights of the dictionary: https://baulab.us/u/smarks/autoencoders/
2. The probe model: https://drive.google.com/file/d/1ULZHKBnJep9IkmgpXUlq_JbKsz2VdiWV/view?usp=sharing
   * The probe model contains the weights of the model trained by the Dr. Sam paper on ambigous dataset.

These are the arguments that needs to be passed while running the `main.py` file. 

```
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
```

```
pip install requirements.txt
python main.py -dict -dr "path" -btr 1 -bts 1 -d "cuda:0" -layer [3,15]
```

## File Information:

`imports.py`: It contains the list of imported modules

`config.py`: It contains the information about different hyperparameters.

`main.py`: It contain the execution taking support from other `.py` files. 


**NOTE**: If it asks for wandb api key then please give "b32f3adb2a2777987a27a6622e4226d85ef8c521".