import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
from sklearn.datasets import make_blobs

#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import torch
import torchvision.models as models
import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image 
from torchvision import models, transforms
import torch.nn as nn
from nct_pickle import NCT_PICKLE


# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
# x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], 
#                   random_state =9)
# plt.scatter(X[:, 0], X[:, 1], marker='o')
# plt.show()
#nct_unlabeled   = NCT_PICKLE("/home/jingyi/ACFSL/nct_pickle", train=True,  transform=moco_transform)
features= torch.load("/home/jingyi/ACFSL/nct_dataset_tif/data_png/features.pth")

x=features.cpu().numpy()
#num_cluster=9
#num_cluster=20
#num_cluster=30
num_cluster=15
y_pred = KMeans(n_clusters=num_cluster, random_state=9).fit_predict(x)
y_pred=torch.tensor(y_pred)
torch.save(y_pred,"/home/jingyi/ACFSL/nct_dataset_tif/data_png/pseudo_labels_{}_clusters_cp0199.pth".format(num_cluster))
