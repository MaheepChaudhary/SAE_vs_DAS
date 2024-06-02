from imports import *

epochs = 4
lr = 0.01

batch_size_train = 64
batch_size_test = 1
momentum = 0.7
log_interval = 10

device = 'cpu' 
gender_dataset = "LabHC/bias_in_bios"
profession_dict = {'professor' : 21, 'nurse' : 13}
male_prof = 'professor'
female_prof = 'nurse'

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)