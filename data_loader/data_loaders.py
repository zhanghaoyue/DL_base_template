import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.dataset_dsa import DSA_Dataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def dsa_pair_pre(df, params, type='train'):
    if type == 'train':
        par = params['data_loader_train']['args']
    elif type == 'valid':
        par = params['data_loader_valid']['args']
    elif type == 'test':
        par = params['data_loader_valid']['args']

    X_data = DSA_Dataset(df, params)
    loader = BaseDataLoader(X_data,
                            batch_size=par['batch_size'],
                            shuffle=par['shuffle'],
                            num_workers=par['num_workers'],
                            validation_split=par['validation_split'], )

    return loader
