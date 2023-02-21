'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
from __future__ import division 
#from pickle import TRUE
from scipy.stats import entropy
import pandas as pd
import os
from random import choice
import random
import torch
#from torch import use_deterministic_algorithms
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
#import torchvision as TV
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10
from nct import NCT
from load_pickled_nct import NCT_PICKLE
from load_pickled_breakhis import BREAKHIS_PICKLE
from torchvision.utils import save_image

# Utils
#import visdom
from tqdm import tqdm

# Custom
import models.resnet as resnet
from config import *
from data.sampler import SubsetSequentialSampler
from PIL import Image 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
#import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# Python and Numpy Random Seed
random.seed(123) # controls python packages, e.g. random.choice()
np.random.seed(123)

# Torch Random Seed
r_seed=459
torch.manual_seed(r_seed)
torch.cuda.manual_seed(r_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# we can assign the work to the GPU card '1' or '0' or '0,1'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#Define the data augamentation(transformations) for data loader.
train_transform = T.Compose([
    T.Resize([84,84]),
    T.ToTensor(),
])
test_transform = T.Compose([
    
    T.Resize([84,84]), # test set resize has to be consistant with the training set.
    T.ToTensor(),   
])

########## Data loading #############
# Both methods work. Just choose one of them.

##### Method 1: check nct.py as reference #####
#data_test  = NCT("/home/jingyi/LLAL_HISTO/nct_dataset_tif/", train=False,  transform=test_transform)
# data_unlabeled   = NCT("/home/jingyi/LLAL_HISTO/nct_dataset_tif/", train=True,  transform=test_transform)
# data_train = NCT("/home/jingyi/LLAL_HISTO/nct_dataset_tif/", train=True,  transform=train_transform)

##############  Method 2: GENERATE DATASET FROM PICKLED NCT, nct_pickle.py ##########
data_test  = NCT_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/nct_pickle/", train=False,  transform=test_transform)
data_unlabeled   = NCT_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/nct_pickle/", train=True,  transform=test_transform)
data_train = NCT_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/nct_pickle/", train=True,  transform=train_transform)


############ Load breakhis dataset ###########
#data_test  = BREAKHIS_PICKLE("./Active_FSL/breakhis_pickle", train=False,  transform=test_transform)
#data_unlabeled   = BREAKHIS_PICKLE("./Active_FSL/breakhis_pickle", train=True,  transform=test_transform)
#data_train = BREAKHIS_PICKLE("./Active_FSL/breakhis_pickle", train=True,  transform=train_transform)


# Train Utils
iters = 0

###########load pseudo labels###########
pseudo_labels=torch.load("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Generate_Pseudo_Labels/pseudo_labels_checkpoint0199.pth")
pseudo_labels=pseudo_labels.cpu().detach().numpy()

####### Get TSNE graphs ###################
def gen_features(dataloaders):
    models['backbone'].eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloaders):
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets_np = targets.data.cpu().numpy()

            outputs, _ = models['backbone'](inputs)
            outputs_np = outputs.data.cpu().numpy()
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(dataloaders)):
                print(idx+1, '/', len(dataloaders))

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, outputs

def tsne_plot(save_dir,  first_selection_K_samples, rest_labeled_samples, targets_un, outputs_un):
    print('generating t-SNE plot...')
    tsne = TSNE(n_iter=5000, init='pca',  perplexity=28, random_state=0) 
    
    tsne_output_un = tsne.fit_transform(outputs_un)
    df_un = pd.DataFrame(tsne_output_un, columns=['x', 'y']) # pd.DataFrame(data, columns 行)
    df_un['unlabeled set'] = targets_un # labels, so df_un has three columns: ['x', 'y','targets']
    
    tsne_output_rest=tsne_output_un[rest_labeled_samples,:]
    df_rest = pd.DataFrame(tsne_output_rest, columns=['x', 'y'])
    df_rest['previous cycles'] = targets_un[rest_labeled_samples]
    
    tsne_output_k=tsne_output_un[first_selection_K_samples,:]
    df_k = pd.DataFrame(tsne_output_k, columns=['x', 'y'])
    df_k['current cycles'] = targets_un[first_selection_K_samples]
    
    plt.figure(figsize=(10,10))
    scatter=sns.scatterplot(
        x='x', y='y', # xlabel data= df_un[x], ylabel data= df_un[y]
        hue='unlabeled set', # Grouping variable that will produce points with different colors 
        hue_order=list(range(0,NUM_CLASSES)),
        palette=sns.color_palette('pastel',NUM_CLASSES),
        data=df_un, #data
        style='unlabeled set',
        style_order=list(range(0,NUM_CLASSES)),

        markers=["."]*NUM_CLASSES,      
     )

    scatter=sns.scatterplot(
        x='x', y='y',
        hue='previous cycles',
        hue_order=list(range(0,NUM_CLASSES)),
        palette=sns.color_palette('deep',NUM_CLASSES),
        data=df_rest,
        style='previous cycles',
        style_order=list(range(0,NUM_CLASSES)),
        markers=["o"]*NUM_CLASSES,

    )
    scatter=sns.scatterplot(
        x='x', y='y',
        hue='current cycles',
        palette=sns.color_palette('deep',NUM_CLASSES),
        data=df_k,
        hue_order=list(range(0,NUM_CLASSES)),
        style='current cycles',
        style_order=list(range(0,NUM_CLASSES)),
        markers=["D"]*NUM_CLASSES,
        #legend='full',
        #alpha=0.8
    )

    scatter.legend(fontsize = 8, 
               bbox_to_anchor= (1.03, 1), 
               title="Sample Type", 
               title_fontsize = 10, 
               shadow = True, 
               facecolor = 'white')
        
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    #plt.show()
    plt.savefig(save_dir, bbox_inches='tight')
    print('done!')


def tsne_plot_everycycle_selection(root_dir,  labeled_set,targets_un, outputs_un):
    print('generating t-SNE plot...')
    tsne = TSNE(n_iter=5000, init='pca',  perplexity=28, random_state=0) # Is this the best TSNE parameter setting?
    tsne_output_un = tsne.fit_transform(outputs_un)
    df_un = pd.DataFrame(tsne_output_un, columns=['x', 'y']) # pd.DataFrame(data, columns 行)
    df_un['unlabeled set'] = targets_un # labels, so df_un has three columns: ['x', 'y','targets']
    for i in range(CYCLES):
    
        first_selection_K_samples=labeled_set[i*ADDENDUM:((i+1)*ADDENDUM)]
        tsne_output_k=tsne_output_un[first_selection_K_samples,:]
        df_k = pd.DataFrame(tsne_output_k, columns=['x', 'y'])
        df_k['current cycles'] = targets_un[first_selection_K_samples]
        
        previous_selected_samples=labeled_set[0:i*ADDENDUM]
        tsne_output_previous=tsne_output_un[previous_selected_samples,:]
        df_previous = pd.DataFrame(tsne_output_previous, columns=['x', 'y'])
        df_previous['previous cycles'] = targets_un[previous_selected_samples]
        
        plt.figure(figsize=(10,10))
        scatter=sns.scatterplot(
            x='x', y='y', # xlabel data= df_un[x], ylabel data= df_un[y]
            hue='unlabeled set', # Grouping variable that will produce points with different colors
            hue_order=list(range(0,NUM_CLASSES)),
            palette=sns.color_palette('pastel',NUM_CLASSES),
            data=df_un, #data
            style='unlabeled set',
            style_order=list(range(0,NUM_CLASSES)),

            markers=["."]*NUM_CLASSES,      
            )
        ########Plot previous cycles selected samples , if no need, just comment this
        scatter=sns.scatterplot(
        x='x', y='y',
        hue='previous cycles',
        palette=sns.color_palette('deep',NUM_CLASSES),
        data=df_previous,
        hue_order=list(range(0,NUM_CLASSES)),
        style='previous cycles',
        style_order=list(range(0,NUM_CLASSES)),
        markers=["o"]*NUM_CLASSES,    
        )
    
        scatter=sns.scatterplot(
            x='x', y='y',
            hue='current cycles',
            palette=sns.color_palette('deep',NUM_CLASSES),
            data=df_k,
            hue_order=list(range(0,NUM_CLASSES)),
            style='current cycles',
            style_order=list(range(0,NUM_CLASSES)),
            markers=["D"]*NUM_CLASSES,
        
        )
    
        scatter.legend(fontsize = 8, 
                bbox_to_anchor= (1.03, 1), 
                title="Sample Type", 
                title_fontsize = 10, 
                shadow = True, 
                facecolor = 'white')
            
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        #plt.show()
        plt.savefig(os.path.join(root_dir,"cycle{}".format(i+1)), bbox_inches='tight')
        print('done!')

##################################################################
###### Training #######
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    
    models['backbone'].train()
    global iters

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        scores, features = models['backbone'](inputs)
        # print(torch.max(scores))
        # print(torch.min(scores))
        target_loss = criterion(scores, labels)
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0) 
        loss=m_backbone_loss
        loss.backward()
        optimizers['backbone'].step()
        
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis=None):

    print('>> Train a Model.')
    best_acc = 0.
    # checkpoint_dir = os.path.join('./nct', 'train', 'weights')
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    for epoch in range(num_epochs):
        #schedulers['backbone'].step()
        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None)
        schedulers['backbone'].step() # perhaps will fix the error
    print('>> Finished.')

####### Test #########
def test(models, dataloaders, mode='val'): 
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    total = 0
    correct = 0
    list_of_classes=list(range(0,NUM_CLASSES,1))
    acc = [0 for c in list_of_classes]
    getacc1= [0 for c in list_of_classes]
    getacc2= [0 for c in list_of_classes]
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()
            scores, _ = models['backbone'](inputs)
            #probs = torch.nn.functional.softmax(scores, dim=1)
            #preds.shape=(batchsize,1) # returns the index of the maximum value of (FC layer output)logits of each sample
            _, preds = torch.max(scores.data, 1) # torch.max returns the result tuple of two output tensors (max, max_indices)
            #_,preds_probs=torch.max(probs.data, 1)
            #subt = torch.sub(preds, preds_probs)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for c in list_of_classes:
                getacc1[c]+=((preds == labels) * (labels == c)).sum().item()
                getacc2[c]+=(labels == c).sum().item()
            
        acc_all=100 * correct / total
        for c in list_of_classes:
            acc[c] = getacc1[c] /max(getacc2[c],1)                
    return acc_all, acc



################# Informativeness calculation methods #################

###### Query strategy: Entropy-based score ############
#def get_uncertainty(models, unlabeled_loader,unlabeled_set):
#    models['backbone'].eval()
#    entropylist=[]
#    with torch.no_grad():
#        for (inputs, labels) in unlabeled_loader:
#            inputs = inputs.cuda()
#            labels = labels.cuda()
#            scores, _ = models['backbone'](inputs)
#            probs = torch.nn.functional.softmax(scores, dim=1)
#            for x in probs:
#                entropylist.append(Categorical(probs = x).entropy())
#        entropylist=torch.Tensor(entropylist)
#    return entropylist

########## Query strategy: Margin sampling ####################
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    uncertainty_score=[]    
    with torch.no_grad():
        i=0
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            scores, _ = models['backbone'](inputs)
            i=i+1
            probs = torch.nn.functional.softmax(scores, dim=1)           
            for x in probs:
                xb=x.sort(0,True)[0]
                marginvalue=xb[0]-xb[1]
                uncertainty_score.append(1.0/marginvalue)                
        uncertainty_score=torch.Tensor(uncertainty_score)
    return uncertainty_score

########## Query strategy: Least confidence####################
# def get_uncertainty(models, unlabeled_loader,unlabeled_set):
#     models['backbone'].eval()
#     uncertainty_score=[]
#     with torch.no_grad():
#         i=0
#         for (inputs, labels) in unlabeled_loader:
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#             scores, _ = models['backbone'](inputs)            
#             i=i+1
#             probs = torch.nn.functional.softmax(scores, dim=1)           
#             for x in probs:
#                 xb=x.sort(0,True)[0]               
#                 uncertainty_score.append(1.0/xb[0])                
#         uncertainty_score=torch.Tensor(uncertainty_score)
#     return uncertainty_score
####################################################################

######### Main #########
if __name__ == '__main__':
    
    y = [];
    x = [i*9 for i in range(CYCLES)]
    
    for trial in range(TRIALS): ## TRIALS=1
        indices = list(range(NUM_TRAIN)) # in config.py, we defined NUM_TRAIN = 10000

##### With no initial set#####
        labeled_set=[]
        first_selection_labeled_set=[]

        unlabeled_set = list(set(indices).difference(set(labeled_set)))   # with no labelled set, this is just list(set(indices))
        random_querylist=torch.randperm(NUM_TRAIN) # create a random index list instead of the query list.
        
        # this stuff is all predone
        
        train_loader = DataLoader(data_train, 
                                    batch_size=NUM_CLASSES, # batchsize of the train_loader is the num_classes
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    pin_memory=True)
        
        test_loader  = DataLoader(data_test, 
                                    batch_size=BATCH # config.py ： BATCH = 128
                                    ) 
        
        # dict for dataloaders

        dataloaders  = {'train': train_loader, 'test': test_loader}
            
        ####### Load Pretrained FSL Model######################
        
        #using resnet 18
        resnet18    = resnet.resnet18(num_classes=NUM_CLASSES).cuda()
        
        # load the weights from the pretrained model
        state_dict = torch.load('/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Pretrain_FSL_Model/model_best_standard.pth.tar') #load weights
        pretrained_dict=state_dict['state_dict']
        model_dict=resnet18.state_dict()
        
        # 1. do not import the weight of FC layers(classifier),as we need to train our own classifier on the target dataset samples.
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        resnet18.load_state_dict(model_dict)
        models = {'backbone': resnet18}
        #Before finetuning the classifier, first we test it on the test set to get the original test accuracy.
        acc_all,acc = test(models, dataloaders, mode='test')
        print("original model test acc =",acc)
        for i in range(NUM_CLASSES):
            print("Class{}_acc:{}".format(i,acc[i]))
            
            
        ############
        ############ ASSUME that all of the above works correctly to load the images in and setup the model
        ############
        

        ################# Active learning cycles################
        for cycle in range(CYCLES): # cycles = 15 in config

            criterion      = nn.CrossEntropyLoss(reduction='none')            
            
            ###### Just finetune classifier in each AL cycle.#####
            # finetuning = retraining the pretrained layers
            
            optim_backbone = optim.SGD(models['backbone'].fc.parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}

            # Training and test
            # assume that the below training functions are correct
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis=None)
            acc_all, acc = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc_all))
            for i in range(NUM_CLASSES):
                print("Class{}_acc:{}".format(i,acc[i]))
            
            y.append(acc_all)
            
            # the training and test portion of the loop is now complete
            # what follows is the active learning portion
    
     
            # Create unlabeled dataloader for the unlabeled unlabeled_set
           
            unlabeled_loader = DataLoader(data_unlabeled, 
                                            batch_size=BATCH, # In config.py, BATCH=128
                                            sampler=SubsetSequentialSampler(unlabeled_set), 
                                            pin_memory=True)
           
            ###### initial unlabeled dataloader with NUM_TRAIN images in it. For TSNE PLOT. #####
            initial_unlabeled_loader=DataLoader(data_unlabeled, 
                                          batch_size=BATCH, 
                                          sampler=SubsetSequentialSampler(indices), 
                                          pin_memory=True)
            
            ###### Measure uncertainty of each sample in the unlabeled_set #####
            uncertainty = get_uncertainty(models, unlabeled_loader)
            #uncertainty = get_uncertainty(models, unlabeled_loader, unlabeled_set) #for entropy
            
            ###### Index in ascending order #####
            arg = np.argsort(uncertainty)
            
            
            ##### First round selection is to select K samples with the highest Query Score.#####
            
            # ADDENDUM most informative samples method
            #first_selection_K_samples=list(torch.tensor(unlabeled_set)[arg][-ADDENDUM:].numpy()) # ADDENDUM=K, select K samples in each active learning cycle.
            
            # even selection method
            step = len(unlabeled_set)//ADDENDUM + 1
            first_selection_K_samples=list(torch.tensor(unlabeled_set)[arg][::step].numpy())
            
            ### Print the cover ###
            first_selection_K_samples_labels=[]
            for i in first_selection_K_samples:
                first_selection_K_samples_labels.append(data_train.targets[i])
            first_selection_K_samples_distribution=Counter(first_selection_K_samples_labels)
            
            selected_p_labels=[]
            # pseudo labels are pre generated labels (!) from another algorithm (?)
            for sample_index in first_selection_K_samples:
                selected_p_labels.append(pseudo_labels[sample_index])
            #print("First selection indices:",first_selection_K_samples)
            print("First selection labels:         ",first_selection_K_samples_labels)
            print("First selection pseudo labels:  ",selected_p_labels)
            print("First selection distribution:   ",first_selection_K_samples_distribution)
            
            
            ##### Get NUM_SHOTS true complete sets according to the query list #########
            list0=[]
            second_selection_samples=[]
            for item in arg: # arg is the query list
                true_label=data_unlabeled.targets[item]
                if list0.count(true_label)<NUM_SHOTS:
                    list0.append(true_label)
                    second_selection_samples.append(item)
                if len(list0)>(NUM_CLASSES*NUM_SHOTS-1): # sample a complete 9-way one-shot support set
                #if len(list0)>(NUM_CLASSES-5): # sample a 5-way one-shot support set
                    break
            #print("second_selection_samples_indices:",second_selection_samples)
            second_selection_samples_labels=[]
            for i in second_selection_samples:
                second_selection_samples_labels.append(data_train.targets[i])
            second_selection_samples_distribution=Counter(second_selection_samples_labels)
            print("second_selection_samples true labels:             ",second_selection_samples_labels)
            print("second_selection_samples true label distribution: ",second_selection_samples_distribution)
                 
            ##### Print statistics #####
            second_selection_samples_labels=[]
            for i in second_selection_samples:
                second_selection_samples_labels.append(data_train.targets[i])
            second_selection_samples_distribution=Counter(second_selection_samples_labels)
            second_selected_p_labels=[]
            for sample_index in second_selection_samples:
                second_selected_p_labels.append(pseudo_labels[sample_index])
            #print("Second selection indices:",second_selection_samples)
            print("Second selection true labels:   ",second_selection_samples_labels)
            print("Second selection pseudo labels: ",second_selected_p_labels)
            print("Second selection distribution:  ",second_selection_samples_distribution)
            
            
            ########## Update the labeled dataset and the unlabeled dataset, respectively #####
            ##### If there's only one-round selection #####         
            ##first_selection_labeled_set += first_selection_K_samples  ### for TSNE PLOT
            labeled_set += first_selection_K_samples 
            
            ##### If there's a second round selection#####          
            # labeled_set += second_selection_samples 
            ##print labeled samples indexes
            ##print("Labeled set sample_indexes: ",labeled_set)
            ###print statistics###
            labeled_set_labels=[]
            for i in labeled_set:
                labeled_set_labels.append(data_train.targets[i])
            labeled_set_distribution=Counter(labeled_set_labels)
            print("The entire labeled set distribution: ",labeled_set_distribution)
            
            ###### Update the unlabeled set #######
            unlabeled_set = list(set(indices).difference(set(labeled_set))) 
            #random_querylist=list(set(random_querylist).difference(set(labeled_set))) 
            
            ###### Update the train_loader #######
            dataloaders['train'] = DataLoader(data_train,
                                            batch_size=NUM_CLASSES, 
                                            sampler=SubsetSequentialSampler(labeled_set), 
                                            pin_memory=True)
            
            first_selection_samples_loader = DataLoader(data_train,
                                            batch_size=NUM_CLASSES, 
                                            sampler=SubsetSequentialSampler(first_selection_K_samples), 
                                            pin_memory=True)
            
            second_selection_samples_loader = DataLoader(data_train,
                                            batch_size=NUM_CLASSES, 
                                            sampler=SubsetSequentialSampler(second_selection_samples), 
                                            pin_memory=True)
            
            
            
            ###### Save images in the second round selection #####
            dir_name="resize84_visualize_labeled_set/Trial{}/{}_cluster/Cycle{}/second_selection/".format(trial+1,NUM_CLUSTERS,cycle+1)
            if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)
                     
            second_selection_data_all=torch.Tensor()
            second_selection_target_all=torch.Tensor()

            for batch_index, data_target in enumerate(second_selection_samples_loader): 
        
                second_selection_data=data_target[0]
                second_selection_target=data_target[1]
                second_selection_data_all=torch.cat((second_selection_data_all,second_selection_data),0)
                second_selection_target_all=torch.cat((second_selection_target_all,second_selection_target),0)
                
            #for i in range(second_selection_data_all.shape[0]):
            #        save_image(second_selection_data_all[i],"resize84_visualize_labeled_set/Trial{}/{}_cluster/Cycle{}/second_selection/img{}_Class{}.png".format(trial+1,NUM_CLUSTERS,cycle+1,i+1,int(second_selection_target_all[i])))
            ########################################################### 
            
        # Save a checkpoint
        # torch.save({
        #             'trial': trial + 1,
        #             'state_dict_backbone': models['backbone'].state_dict(),
        #             #'state_dict_module': models['module'].state_dict()
        #         },
        #         './nct/train/weights/active_resnet18_nct_trial{}.pth'.format(trial))
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Labelled Examples")
    plt.ylabel("Test Accuracy")
    plt.savefig('img/acc_graph.png')