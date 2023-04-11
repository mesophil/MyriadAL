import torch
import torchvision.models as models
import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image 
from torchvision import models, transforms
import torch.nn as nn
from nct_pickle import NCT_PICKLE
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
from sklearn.datasets import make_blobs

#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


def extractor( x, net, use_gpu):
   
    
    # img = Image.open(img_path)
    # #img=torch.Tensor(img)
    # img = transform(img)
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x)
    #y = y.data.cpu().numpy()
    #np.savetxt(saved_path, y, delimiter=',')
    return y
  
###导入模型，加载权重####################
if __name__ == '__main__':

    checkpoint = torch.load('/home/jingyi/ACFSL/ACFSL_train_encoder/moco/checkpoint_0199.pth.tar',map_location="cpu")
    arch = checkpoint['arch']
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch]()
    # load from pre-trained, before DistributedDataParallel constructor
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print("=> loaded pre-trained model ")
    model = torch.nn.DataParallel(model).cuda()
    #print(model.state_dict().keys())
    ##删除mlp head或fc层，只留下model的encoder部分
    del model.module.fc
    #print(model.state_dict().keys())

    ##########################################################
    ################load data##############
    # #读取目标路径：/home/jingyi/ACFSL/nct_dataset_tif/data_png/train_test 下的9个文件夹中的50000张图片数据
    # data_dir = '/home/jingyi/ACFSL/nct_dataset_tif/data_png/train_test'
    # #a=data_dir
    # features_dir = '/home/jingyi/ACFSL/nct_dataset_tif/data_png/features'
    # #shutil.copytree(data_dir, os.path.join(features_dir, data_dir))
    # extensions = ['jpg','png', 'jpeg', 'JPG', 'JPEG']
        
    # files_list = []
    # sub_dirs = [x[0] for x in os.walk(data_dir) ]
    # sub_dirs = sub_dirs[1:]
    # for sub_dir in sub_dirs:
    #     for extention in extensions:
    #         file_glob = os.path.join(sub_dir, '*.' + extention)
    #         files_list.extend(glob.glob(file_glob))
    
    
    moco_transform = transforms.Compose([
        #transforms.Scale(256),
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),  ##和MOCO 训练时一致的resize和normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    
    nct_unlabeled   = NCT_PICKLE("/home/jingyi/ACFSL/nct_pickle", train=True,  transform=moco_transform)
    
    unlabeled_samples=torch.tensor(nct_unlabeled.data.transpose((0,3,1,2)))
            
            
    ############提取 所有unlabeled sample 的特征，并且保存起来。#################
    resnet50_feature_extractor=model
    #   resnet50_feature_extractor = models.resnet50(pretrained = True)
    #   resnet50_feature_extractor.fc = nn.Linear(2048, 2048)
    #   torch.nn.init.eye(resnet50_feature_extractor.fc.weight)
    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False  

    use_gpu = torch.cuda.is_available()
    features=torch.Tensor().cuda()
    i=0
    for x in unlabeled_samples:
        y=extractor( x, resnet50_feature_extractor, use_gpu)
        features=torch.cat((features,y),0)  
        print(i)
        i+=1
    torch.save(features,"/home/jingyi/ACFSL/nct_dataset_tif/data_png/features.pth")
    #testfeatures = torch.load("/home/jingyi/ACFSL/nct_dataset_tif/data_png/features.pth")
    # print(testfeatures)


    #########k-means########
    x=features.cpu().numpy()
    y_pred = KMeans(n_clusters=9, random_state=9).fit_predict(x)
    gt_labels=nct_unlabeled.targets  ##list
    gt_labels=np.array(gt_labels)
    b=(y_pred==gt_labels).sum().item()
    pseudo_label_acc=b/len(gt_labels)
    print("pseudo_label_acc=",pseudo_label_acc)
    
    y_pred=torch.tensor(y_pred)
    torch.save(y_pred,"/home/jingyi/ACFSL/nct_dataset_tif/data_png/pseudo_labels_checkpoint0199.pth")

    
    
    
    wait=0