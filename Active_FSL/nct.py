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

class NCT(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            exists 
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            #download: bool = False,
    ) -> None:
        super(NCT, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        ##########
        #root = "/home/jingyi/LLAL_HISTO/nct_dataset_tif/"
        if self.train:
            foldName = "train_png"            
        else:
            foldName = "test_png"
        
        self.data: Any = []
        self.targets = []

        self.data, self.targets = self.load_Img(root,foldName)
        
       
        #self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
    def str2num(self,s): # 将标签名称转换成数字
        digits = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4,'MUS': 5,'NORM': 6,'STR': 7,'TUM': 8}
        return digits[s]
   
            
        '''
        Load the image files form the folder
        input:
        imgDir: the direction of the folder
        imgName:the name of the folder
        output:
        data:the data of the dataset
        label:the label of the datset
        '''
    def load_Img(self,imgDir:str,imgFoldName:str) -> Tuple [Any, Any]:
        imgs = os.listdir(imgDir+imgFoldName)
        imgNum = len(imgs)
        data = np.empty((imgNum,224,224,3),dtype="uint8") # 原来是(imgNum,1,12,12)
        #label = np.empty((imgNum,),dtype="uint8")
        label=[None]*imgNum
        img_path= imgDir+imgFoldName+"/"
        for i in range (imgNum): #for 循环，读images并转化成array，shape=(图片数，像素,像素,3)
            #img = Image.open(imgDir+imgFoldName+"/"+imgs[i])
            img = cv2.imread(img_path+imgs[i])
           # img = img.resize((32, 32), Image.ANTIALIAS)     # 在这里 resize了
            data[i,:,:,:] = np.asarray(img,dtype="uint8")
            label[i]=imgs[i].split('-')[0]
            #label[i] = self.str2num(imgs[i].split('-')[0]) #读文件名“_”前面的部分
        label = list(map(self.str2num, label))    # 这里使用自定义的str2num 函数，将label list 中的标签名称转成了数字 
        return data,label

    
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


# train_hd=pd.read_csv('E:\\360downloads\\name.csv')//获取图片的名字我的csv文件储存在这里
# train_path='E:\\360downloads\\train'              //获取图片的路径（只需要读取到储存图片的文件夹就行了）
#class Mydataset(torch.utils.Data):
    
    # def __init__(self, df_data,data_dir='./',transform=None):
    #     super().__init__()
    #     self.df=df_data.values
    #     self.data_dir=data_dir
    #     self.transform=transform
    # def __len__(self):
    #     return len(self.df)
    
    # def __getiem__(self,idex):
    #     img_name,label=self.df[idex]
    #     img_path=os.path.join(self.data_dir,img_name)
    #     image=cv2.imread(img_path)
    #     if self.transform is not None:
    #         image=self.transform(image)
    #     return image,label             