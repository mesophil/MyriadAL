from tkinter import Label

from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset


#from torch.utils.Data import DataLoader,Dataset
import torchvision.transforms as transforms   
import pandas as pd
import cv2    
import os


class BREAKHIS_PICKLE(VisionDataset):
    """
    Args
        root (string): Root directory of dataset where directory
            ``nct-batches-py`` exists.
        train (bool, optional): If True, creates dataset 
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'breakhis-batches-py'
    train_list = [
        'data_batch_1',
        'data_batch_2'
    
    ]

    test_list = ['test_batch']
    # meta = {
    #     'filename': 'batches.meta',
    #     'key': 'label_names',
    # }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            
    ) -> None:

        super(BREAKHIS_PICKLE, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        

        
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            # print(type(file_name))
            # print(type(downloaded_list))

            file_path = os.path.join(self.root, self.base_folder, str(file_name))
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1') #在这儿 upickle
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        #######这里 reshape(-1,3,32,32) 应该要改成(-1, 3, ?,?)
        self.data = np.vstack(self.data).reshape(-1, 3,460, 700) # This is correct.
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # self._load_meta()

    # def _load_meta(self) -> None:
    #     path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        
    #     with open(path, 'rb') as infile:
    #         data = pickle.load(infile, encoding='latin1')
    #         self.classes = data[self.meta['key']]
        #self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    
    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


