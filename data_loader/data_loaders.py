import torch
from torchvision import datasets, transforms, utils
from base import BaseDataLoader
import numpy as np
from data_loader.dataset_sample import WholeslideDataset
from monai.apps.nuclick.transforms import AddLabelAsGuidanced, SetLabelClassd, SplitLabeld
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    RandTorchVisiond,
    ScaleIntensityRangeD,
    SelectItemsd,
    ToTensord,
)


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


class_names = {
            "0": "non",
            "1": "gland",
        }


def train_pre_transforms():
    transforms = Compose(
        [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            EnsureChannelFirstd(keys=("image", "label")),
            SplitLabeld(keys="label", mask_value=None, others_value=255, to_binary_mask=False),
            RandTorchVisiond(
                keys="image",
                name="ColorJitter",
                brightness=64.0 / 255.0,
                contrast=0.75,
                saturation=0.25,
                hue=0.04,
            ),
            RandFlipd(keys=("image", "label", "others"), prob=0.5),
            RandRotate90d(keys=("image", "label", "others"), prob=0.5, spatial_axes=(0, 1)),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            AddLabelAsGuidanced(keys="image", source="label"),
            SetLabelClassd(keys="label", offset=-1),
            SelectItemsd(keys=("image", "label")),
        ]
    )
    return transforms


def train_post_transforms():
    transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=len(class_names)),
            ToTensord(keys=("pred", "label"), device="cuda"),
        ]
    )
    return transforms


def val_transforms():
    transforms = Compose(
        [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            EnsureChannelFirstd(keys=("image", "label")),
            SplitLabeld(keys="label", mask_value=None, others_value=255, to_binary_mask=False),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            AddLabelAsGuidanced(keys="image", source="label"),
            SetLabelClassd(keys="label", offset=-1),
            SelectItemsd(keys=("image", "label")),
        ]
    )
    return transforms


class PandaDataLoader(BaseDataLoader):
    def __init__(self, data_trsfm, dataset, params, train_type='train'):

        self.data_dir = dataset
        if train_type == 'train':
            par = params['data_loader_train']['args']
            data_transform = train_pre_transforms
        elif train_type == 'valid':
            par = params['data_loader_valid']['args']
            data_transform = val_transforms
        elif train_type == 'test':
            par = params['data_loader_valid']['args']
            data_transform = val_transforms
        else:
            raise NotImplementedError
        if not data_trsfm:
            self.trsfm = data_transform
        else:
            self.trsfm = data_trsfm

        self.dataset = WholeslideDataset(dataset, custom_transforms=self.trsfm)
        self.batch_size = par['batch_size']
        self.shuffle = par['shuffle']
        self.num_workers = par['num_workers']
        self.validation_split = par['validation_split']
        super().__init__(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                         validation_split=self.validation_split, num_workers=self.num_workers)

