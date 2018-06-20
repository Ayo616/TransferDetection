import torchvision
import torch
from torchvision import datasets,transforms
import torch.utils.data as data_utils
import configparser

conf = configparser.ConfigParser()
conf.read('config.ini')
conf = conf['S1']


'''
# Define the DataLoader, loading the train data and test data

'''

class Factory():

    def __init__(self):
        import pandas as pd
        train = pd.read_csv('../dataset/traindata.csv')
        test = pd.read_csv('../dataset/testdata.csv')
        train = torch.from_numpy(train.values)
        test = torch.from_numpy(test.values)

        self.train_data = train[:,0:-1]
        self.train_target = train[:,-1:-1]
        self.test_data = test[:,0:-1]
        self.test_target = test[:,-1:-1]

        extra_test = pd.read_csv('../dataset/pureoutlierdata.csv')
        self.extra_test_data = test[:,0:-1]
        self.extra_test_target = test[:,-1:-1]


    def load_data(self,feature_tensor,label_tensor):
        torch_dataset = data_utils.TensorDataset(feature_tensor,label_tensor)
        loader = data_utils.DataLoader(
            dataset=torch_dataset,
            batch_size=int(conf.get('BATCH_SIZE')),
            shuffle=True,
            num_workers=1,
        )
        return loader

    def get_train_data(self):
        return  self.load_data(self.train_data,self.train_target)

    def get_test_data(self):
        return  self.load_data(self.test_data,self.test_target)

    def get_extra_test_data(self):
        return self.load_data(self.extra_test_data,self.extra_test_target)

if __name__ == '__main__':

    a = Factory().get_train_data()
    a = list(enumerate(a))
    print(type(a))
    for i in range(3):
        print(i)
        co = a[1]
        print(co)
