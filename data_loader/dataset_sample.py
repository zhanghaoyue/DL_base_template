### https://github.com/IPMI-ICNS-UKE/DeepTICI/blob/87f3e22bf096b405a0c016ca8b86b92aa43c9604/data_loader.py

from torch.utils.data.dataset import Dataset
import h5py
import numpy as np


class WholeslideDataset(Dataset):
    def __init__(self,
                 file_path,
                 wsi_slide,
                 wsi_mask,
                 custom_transforms=None,
                 custom_downsample=1,
                 target_patch_size=-1
                 ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.wsi = wsi_slide
        self.wsi_mask = wsi_mask
        self.roi_transforms = custom_transforms
        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample,) * 2
            else:
                self.target_patch_size = None
        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        mask = self.wsi_mask.read_region(coord, self.patch_level, (self.patch_size, self.patch_size))
        mask[mask < 3] = 0
        mask[mask >= 3] = 1
        mask = mask.convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
            mask = mask.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        mask = self.roi_transforms(mask).unsqueeze(0)

        return {"image": img, "label": mask}
