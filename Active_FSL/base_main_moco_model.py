'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
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
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import os
from random import choice
import random
# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

# Torchvison
import torchvision.transforms as T
#import torchvision as TV
import torchvision.models as models
from nct import NCT
from nct_pickle import NCT_PICKLE
#from load_pickled_breakhis import BREAKHIS_PICKLE
from torchvision.utils import save_image

# Utils
#import visdom
from tqdm import tqdm

# Custom
import models.resnet as resnet
from config import *
from data.sampler import SubsetSequentialSampler
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt



os.environ["CUDA_VISIBLE_DEVICES"] = '0,1' # use GPU 1.

#### settings ######
r_seed=123
####################

##### Reproducibility #####
# Python and Numpy Random Seed
random.seed(123) # It controls python packages, e.g. random.choice()
np.random.seed(123)
# Torch Random Seed
torch.manual_seed(r_seed)
torch.cuda.manual_seed(r_seed)

#torch.backends.cudnn.deterministic = True # It applies to CUDA convolution operations for reproducibility.
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'  #  if you are using CUDA tensors, and your CUDA version is 10.2 or greater, you should set the environment variable CUBLAS_WORKSPACE_CONFIG according to CUDA documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
torch.set_deterministic(True) # It affects all the normally-nondeterministic operations for reproducibility.
torch.backends.cudnn.benchmark = False #It causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.

# we can assign the work to the GPU card '1' or '0' or '0,1'. When '0,1', it uses GPU 0 first, if not enough, then use GPU 0 and 1.
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
##############################

#Define the data augamentation(transformations) for data loader.
# train_transform = T.Compose([

#     T.Resize([84,84]), # We resized the source dataset images to 84*84 pixels while pretraining the FSL model, so here we must resize training set and test set to the same image size.
#     T.ToTensor(), # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
    
#     # #T.Normalize([0.7051, 0.5320, 0.7401], [0.1574, 0.2173, 0.1652])
# ])
# test_transform = T.Compose([
#     T.Resize([84,84]), # test set resize has to be consistant with the training set.
#     T.ToTensor(),   
#     #T.Normalize([0.7051, 0.5320, 0.7401], [0.1574, 0.2173, 0.1652])
# ])
moco_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),  ##和MOCO 训练时一致的resize和normalization
            # transforms.Normalize(
            # mean=[0.7401, 0.5320, 0.7051], 
            # std=[0.1281, 0.1608, 0.1192]) #NCT DATASET
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # IMAGETNET DATASET
            ])

########## Data loading #############

data_test  = NCT_PICKLE("/home/jingyi/ACFSL/nct_pickle", train=False,  transform=moco_transform)
data_unlabeled   = NCT_PICKLE("/home/jingyi/ACFSL/nct_pickle", train=True,  transform=moco_transform)
data_train = NCT_PICKLE("/home/jingyi/ACFSL/nct_pickle", train=True,  transform=moco_transform)


############ Load breakhis dataset ###########

# data_test  = BREAKHIS_PICKLE("/home/jingyi/pickle_breakhis/breakhis_pickle", train=False,  transform=test_transform)
# data_unlabeled   = BREAKHIS_PICKLE("/home/jingyi/pickle_breakhis/breakhis_pickle", train=True,  transform=test_transform)
# data_train = BREAKHIS_PICKLE("/home/jingyi/pickle_breakhis/breakhis_pickle", train=True,  transform=train_transform)



###########load first-generation pseudo labels###########
pseudo_labels=torch.load("/home/jingyi/ACFSL/nct_dataset_tif/data_png/pseudo_labels_checkpoint0199.pth") # NCT pseudo labels by moco+kmeans
pseudo_labels=pseudo_labels.cpu().detach().numpy()
   
###### Training #######
def train(models, criterion, optimizers, schedulers,dataloaders,num_epochs):
    print('>> Train a Model.')
    models.train() # Set the model to training mode
    # Iterate over the training set for a few epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        schedulers.step()
        for data in tqdm(dataloaders['train'], leave=False):
            # Move the inputs and labels to the GPU
            inputs = data[0].cuda()
            labels = data[1].cuda()

            # Zero the gradients
            optimizers.zero_grad() #In this for loop if we do not set the optimizer to zero every time the past value it may get add up and changes the result. So we use zero_grad to not face the wrong accumulated results.
            # Make predictions
            #scores, _ = models(inputs)
            scores= models(inputs)

            target_loss = criterion(scores, labels)
            loss = torch.sum(target_loss) / target_loss.size(0) 
            running_loss += loss.item()
            # Backpropagate the gradients
            loss.backward()
            # Update the parameters
            optimizers.step()
            ####To obeserve training loss####
        #print("Epoch {} average loss: {:.4f}".format(epoch, running_loss / len(dataloaders['train'])))
    
    print('>> Finished.')

####### Test #########
def test(models, dataloaders, mode='val'): 
    #assert mode == 'val' or mode == 'test'
    models.eval()
    total = 0
    correct = 0
    list_of_classes=list(range(0,NUM_CLASSES,1))
    acc = [0 for c in list_of_classes]
    getacc1= [0 for c in list_of_classes]
    getacc2= [0 for c in list_of_classes]
    
    pred_labels= torch.Tensor()     # pred_labels will save the predicted labels of all samples.
    pred_labels = pred_labels.type(torch.int64)
    true_labels= torch.Tensor()     # true_labels will save the true labels of all samples.
    true_labels = true_labels.type(torch.int64)    
    with torch.no_grad():
        for (inputs, labels,fnames) in dataloaders[mode]: #fnames get the .png file names. 
            inputs = inputs.cuda()
            labels = labels.cuda()
            pred_labels=pred_labels.cuda()
            true_labels=true_labels.cuda()
            scores= models(inputs)                   # score is the predicted possibility of each class for the specific sample            

            #scores, _ = models(inputs)                   # score is the predicted possibility of each class for the specific sample            
            _, preds = torch.max(scores.data, 1) # torch.max returns the result tuple of two output tensors (max, max_indices)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for c in list_of_classes:
                getacc1[c]+=((preds == labels) * (labels == c)).sum().item()
                getacc2[c]+=(labels == c).sum().item()
            true_labels=torch.cat((true_labels,labels),0)   # pred_labels will save the predicted labels of all samples.         
            pred_labels=torch.cat((pred_labels,preds),0)    # true_labels will save the true labels of all samples.
    acc_all=100 * correct / total
    for c in list_of_classes:
        acc[c] = getacc1[c] /max(getacc2[c],1)                
    return acc_all, acc, true_labels, pred_labels



######### Main #########
if __name__ == '__main__':
    
    ##### Save all printed content to log.txt file. #####
    dir_name="/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/logs/"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    log = open(dir_name+"replace_fsl_with_moco_fixed_plabels_log.txt", mode='a',encoding='utf-8')

    
    ##### Create lists of labeled_set, unlabeled_set, containing sample indices. #####
    indices = list(range(NUM_UNLABELED))# NUM_UNLABELED is the number of samples in unlabeled set at the very beginning.                            # e.g. There are 50000 samples in unlabeled set, but we can set NUM_UNLABELED=10000 to use just the first 10000 samples.
    labeled_set=[] # With no initial labeled set
    unlabeled_set = indices   

    random_querylist_0 = list(range(NUM_UNLABELED))
    random.shuffle(random_querylist_0)
    random_querylist=random_querylist_0
    
    
    ##### Create dataloaders #####
    train_loader = DataLoader(data_train, 
                                batch_size=NUM_CLASSES, # batchsize of the train_loader is the num_classes
                                sampler=SubsetSequentialSampler(labeled_set), 
                                pin_memory=True)
    
    test_loader  = DataLoader(data_test, 
                                batch_size=BATCH
                                ) 
    all_loader = DataLoader(data_unlabeled, 
                                batch_size=BATCH, # In config.py, BATCH=128
                                sampler=SubsetSequentialSampler(indices), 
                                pin_memory=True)
    dataloaders  = {'train': train_loader, 'test': test_loader,'all':all_loader}
    
    
    ####### Load Pretrained MOCO Model(Encoder)######################
    
    checkpoint = torch.load('/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Pretrain_FSL_Model/best_pretrain_moco_encoder_k=65536_checkpoint_0199.pth.tar',map_location="cpu")
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
        
    model.fc = nn.Linear(2048, NUM_CLASSES)
    #Xavier initialization
    torch.nn.init.xavier_uniform_(model.fc.weight)
    model.load_state_dict(state_dict, strict=False)
    print("=> loaded pre-trained model ")
    models = model.cuda()

    ################# Active learning cycles################
    for cycle in range(CYCLES):
        criterion = nn.CrossEntropyLoss(reduction='none')
        ###### Uncomment this part to finetune whole model in each AL cycle.######
        # for param in models.parameters():
        #     param.requires_grad = True
        # models = models.cuda()
        # # for k,v in models.named_parameters():
        # #     print('{}: {}'.format(k, v.requires_grad))
        # optimizers = optim.SGD(models.parameters(), lr=LR, 
        #                         momentum=MOMENTUM, weight_decay=WDECAY)
        
        ###### Just finetune classifier in each AL cycle.#####
        optimizers = optim.Adam(models.fc.parameters(), lr=LR, 
                                 weight_decay=WDECAY)
        schedulers = lr_scheduler.MultiStepLR(optimizers, milestones=MILESTONES)
        
        ####### Training and test##########
        if cycle>0: # In the first cycle, the labeled set is empty, so we don't train the model. We start training the model at the second AL cycle.
            
            #finetune the moco classifier
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH)
            #Update pseudo labels here
            acc_all_unlabeled,acc_unlabeled,true_labels_unlabeled,pseudo_labels = test(models, dataloaders, mode='all')
            torch.save(pseudo_labels,"/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/p_labels/pseudo_labels_cycle{}.pth".format(cycle+1))
            print('Cycle {}/{} || Label set size {}: unlabeled set acc {}'.format(cycle+1, CYCLES, len(labeled_set), acc_all_unlabeled),file = log)
            print('Cycle {}/{} || Label set size {}: unlabeled set acc {}'.format(cycle+1, CYCLES, len(labeled_set), acc_all_unlabeled))
            for i in range(NUM_CLASSES):
                print("unlabeled set Class{}_acc:{}".format(i,acc_unlabeled[i]),file = log)
                print("unlabeled set Class{}_acc:{}".format(i,acc_unlabeled[i]))
            acc_all,acc,true_labels,pred_labels = test(models, dataloaders, mode='test')       
            ###### Test Result Visualization and Statistics  #####
            print('Cycle {}/{} || Label set size {}: Test set acc {}'.format(cycle+1, CYCLES, len(labeled_set), acc_all),file = log)
            print('Cycle {}/{} || Label set size {}: Test set acc {}'.format( cycle+1, CYCLES, len(labeled_set), acc_all))
            for i in range(NUM_CLASSES):
                print("Class{}_acc:{}".format(i,acc[i]),file = log)
                print("Class{}_acc:{}".format(i,acc[i]))
        
        ##################################################################
        ###### Create unlabeled dataloader for the unlabeled unlabeled_set###############
        unlabeled_loader = DataLoader(data_unlabeled, 
                                        batch_size=BATCH, # In config.py, BATCH=128
                                        sampler=SubsetSequentialSampler(unlabeled_set), 
                                        pin_memory=True)
    
        
      ###########################################################################################
        
        # ####### randomly select one pseudo complete set each cycle######################
        list0=[]
        selected_samples=[]
        for item in random_querylist: # arg_reverse is the query list in decsending order.
            pseudo_label=pseudo_labels[item]
            if list0.count(pseudo_label)<1:
                list0.append(pseudo_label)
                selected_samples.append(item)
            if len(list0)>(NUM_CLASSES-1):
                break
        
        ################################################################################################
        #Just random select K=NUM_CLASSES samples from the unlabeled dataset each AL cycle.
        #without any help of pseudo labels or AL strategies
        #selected_samples = random_querylist[:NUM_CLASSES]

        ##################################################################################################
        
         # ##### randomly select one true complete set each cycle  #########
        # list0=[]  # list0 is used to save the true labels of samples selected this AL cycle.
        # selected_samples=[] #selected_samples is used to save the index of samples selected this AL cycle.
        # for item in random_querylist:
        #     true_label=data_unlabeled.targets[item] # Get the true label of this sample
        #     if list0.count(true_label)<1:  # count how many element=true label in list0
        #         list0.append(true_label)           # if less than 1, select this sample, save its label in list0. 
        #         selected_samples.append(item) # select this sample,save its index in list:selected_samples
        #     if len(list0)>(NUM_CLASSES-1): # sample a complete 9-way one-shot support set
        #     #if len(list0)>(NUM_CLASSES-5): # sample a 5-way one-shot support set
        #         break
        ##################################################################################################
        
        

        
                
        ##### Print statistics #####
        selected_samples_labels=[]
        for i in selected_samples:
            selected_samples_labels.append(data_train.targets[i])
        selected_samples_distribution=Counter(selected_samples_labels)
        selected_p_labels=[]
        for sample_index in selected_samples:
            selected_p_labels.append(pseudo_labels[sample_index])
        integer_selected_p_labels = [x.item() for x in selected_p_labels]

        print(" selected_samples_indices:",selected_samples,file = log)
        print("selected_samples_true_labels: ",selected_samples_labels,file = log)
        
        print("selected_samples_pseuodo_labels: ",integer_selected_p_labels,file = log)
        print("selected_samples_distribution: ",selected_samples_distribution,file = log)

        print("selected_samples_indices:",selected_samples)
        print("selected_samples_true_labels: ",selected_samples_labels)
        print("selected_samples_pseuodo_labels: ",integer_selected_p_labels)
        print("selected_samples_distribution: ",selected_samples_distribution)            
       
        
    
        ########## Update the labeled dataset and the unlabeled dataset, respectively #####
        
        labeled_set += selected_samples 
        labeled_set_labels=[]
        for i in labeled_set:
            labeled_set_labels.append(data_train.targets[i])
        labeled_set_distribution=Counter(labeled_set_labels)
        #integer_labeled_set = [x.item() for x in labeled_set]
        print("The entire labeled set: ",labeled_set,file = log)
        print("The entire labeled set: ",labeled_set)
        print("The entire labeled set distribution: ",labeled_set_distribution,file = log)
        print("The entire labeled set distribution: ",labeled_set_distribution)
        
         ###### Update the unlabeled set #######
        unlabeled_set = [i for i in indices if i not in labeled_set]
        random_querylist=[i for i in random_querylist_0 if i not in labeled_set]
        #print(random_querylist[0:50])
        ###### Update the train_loader #######
        dataloaders['train'] = DataLoader(data_train,
                                        batch_size=NUM_CLASSES, 
                                       sampler=SubsetSequentialSampler(labeled_set), 
                                        pin_memory=True)
        

    log.close()