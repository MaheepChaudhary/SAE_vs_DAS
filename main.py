from imports import *









argparser = argparse.ArgumentParser()
argparser.add_argument('-dict','download_dictionary', type=bool, help="a boolean, helping to get you dictionary if you don't have it.")
argparser.add_argument('data_dir', type=str, help='directory to save the data')
argparser.add_argument('-e','--epochs', default=15, type=int, help='number of epochs')
argparser.add_argument('-lr','--lr', default=0.001, type=float, help='learning rate')
argparser.add_argument('-btr','batch_size_train', type=int, help='batch size for training')
argparser.add_argument('-bts','batch_size_test', type=int, help='batch size for testing')
argparser.add_argument("-d",'device', type=str, help='device to be used')
argparser.add_argument("-layer",'residual layer', type=str, help="residual layer to be used interevened in the model")
args = argparser.parse_args()

