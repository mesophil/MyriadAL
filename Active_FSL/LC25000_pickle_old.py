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


class LC25000_PICKLE(VisionDataset):
    """
    Args
        root (string): Root directory of dataset where directory
            ``breakhis-batches-py`` exists.
        train (bool, optional): If True, creates dataset 
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        

    """

###############
    base_folder = 'LC25000-batches-py'
    train_list = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4'
    
    ]

    test_list = ['test_batch']
##########################
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            
    ) -> None:

        super(LC25000_PICKLE, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        self.filenames = []


        # now load the picked numpy arrays
        for file_name in downloaded_list:
            # print(type(file_name))
            # print(type(downloaded_list))

            file_path = os.path.join(self.root, self.base_folder, str(file_name))
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1') # upickle
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                    
                if 'filenames' in entry:
                   
                    self.filenames.extend(entry['filenames'])

        #######CIFAR10: reshape(-1,3,32,32) 
        self.data = np.vstack(self.data).reshape(-1, 3,768, 768) # LC25000 image size is 768 × 768 pixels
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:s
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, filename = self.data[index], self.targets[index], self.filenames[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image, array-> PIL image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, filename

    def __len__(self) -> int:
        return len(self.data)