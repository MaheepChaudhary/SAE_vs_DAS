from imports import *

epochs = 4
lr = 0.01

batch_size_train = 64
batch_size_test = 1
momentum = 0.7
log_interval = 10

device = 'cpu' 

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)