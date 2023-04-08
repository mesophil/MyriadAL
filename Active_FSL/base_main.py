from __future__ import division 
from scipy.stats import entropy
import pandas as pd
import os
from random import choice
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
from config import *

import torch.hub
from torchvision.models import get_model_weights, get_weight

# Torchvison
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10
from nct import NCT
from load_pickled_nct import NCT_PICKLE
from LC25000_pickle import LC25000_PICKLE
from load_pickled_breakhis import BREAKHIS_PICKLE
from torchvision.utils import save_image

# Utils
from tqdm import tqdm

# Custom
import models
import models.resnet as resnet
from data.sampler import SubsetSequentialSampler
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Python and Numpy Random Seed
random.seed(123) # controls python packages, e.g. random.choice()
np.random.seed(123)

# Torch Random Seed
r_seed=RANDOM_SEED
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

################### Data loading ###################

############## Load pickled NCT ##############
# data_test  = NCT_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/nct_pickle/", train=False,  transform=test_transform)
# data_unlabeled   = NCT_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/nct_pickle/", train=True,  transform=test_transform)
# data_train = NCT_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/nct_pickle/", train=True,  transform=train_transform)


############## Load pickled breakhis dataset ##############
# data_test  = BREAKHIS_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/breakhis_pickle", train=False,  transform=test_transform)
# data_unlabeled   = BREAKHIS_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/breakhis_pickle", train=True,  transform=test_transform)
# data_train = BREAKHIS_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/breakhis_pickle", train=True,  transform=train_transform)


############## load pickled LC25000 dataset ##############
data_test  = LC25000_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/LC25000_pickle", train=False,  transform=test_transform)
data_unlabeled   = LC25000_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/LC25000_pickle", train=True,  transform=test_transform)
data_train = LC25000_PICKLE("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Active_FSL/LC25000_pickle", train=True,  transform=train_transform)

############## load pseudo labels ##############
pseudo_labels=torch.load("/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Generate_Pseudo_Labels/breakhis_pseudo_labels_checkpoint0199.pth")
pseudo_labels=pseudo_labels.cpu().detach().numpy()


################### Functions ###################

############## Training ##############

# add parameter for the double train
# in loop do something like if PARAM == 2 or i % 2 == PARAM - use 2 for full set, 1 for first subset, 0 for second subset
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, sub=2):
    
    models['backbone'].train()
    i = 0
    # was dataloaders['train']
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        if i % 2 == sub or sub == 2:
            inputs = data[0].cuda()
            labels = data[1].cuda()
            optimizers['backbone'].zero_grad()
            scores, features = models['backbone'](inputs)
            target_loss = criterion(scores, labels)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0) 
            loss=m_backbone_loss
            loss.backward()
            optimizers['backbone'].step()
        i = i + 1
            
        
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, sub=2):

    print('>> Training...')
    for epoch in range(num_epochs):
        #schedulers['backbone'].step()
        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, sub)
        schedulers['backbone'].step()
    print('>> Finished Training.')

############## Testing ##############
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
        for (inputs, labels, filenames) in dataloaders[mode]:
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


def probs(models, dataloaders, mode='val'):
    """
    models = {'backbone': resnet18}
    dataloaders  = {'train': train_loader, 'test': test_loader}
    
    """
    assert mode == 'val' or mode == 'test'
    
    # switch to evaluation mode
    models['backbone'].eval() 
    
    prob = []
    
    with torch.no_grad(): #speed up computation when no backprop is necessary
        for (inputs, labels, filenames) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()
            logits, _ = models['backbone'](inputs)
            sc = torch.nn.functional.softmax(logits, dim=1)
            prob.append(sc)
    prob = torch.stack(prob)
    return prob




############## Informativeness calculation methods ##############

###### Query strategy: Entropy-based score ######
def get_uncertainty_entropy(models, unlabeled_loader,unlabeled_set):
    models['backbone'].eval()
    entropylist=[]
    with torch.no_grad():
        for (inputs, labels, filenames) in unlabeled_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            scores, _ = models['backbone'](inputs)
            probs = torch.nn.functional.softmax(scores, dim=1)
            for x in probs:
                entropylist.append(Categorical(probs = x).entropy())  # 0 < uncertainty_score < log 1/N
        entropylist=torch.Tensor(entropylist)
    return entropylist

###### Query strategy: Margin sampling ######
def get_uncertainty_margin(models, unlabeled_loader):
    models['backbone'].eval()
    uncertainty_score=[]    
    
    with torch.no_grad():
        for (inputs, labels, filenames) in unlabeled_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            scores, _ = models['backbone'](inputs)
            probs = torch.nn.functional.softmax(scores, dim=1)           
            for x in probs:
                xb=x.sort(0,True)[0]
                marginvalue=xb[0]-xb[1]
                uncertainty_score.append(1.0/marginvalue)      # 1 < uncertainty_score < inf                
        uncertainty_score=torch.Tensor(uncertainty_score)
    return uncertainty_score


###### Query Strategy: Least Confidence ######
def get_uncertainty_LC(models, unlabeled_loader):
    models['backbone'].eval()
    uncertainty_score=[]
    with torch.no_grad():
        for (inputs, labels, filenames) in unlabeled_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            scores, _ = models['backbone'](inputs)        
            probs = torch.nn.functional.softmax(scores, dim=1)           
            for x in probs:
                xb=x.sort(0,True)[0]               
                uncertainty_score.append(1.0/xb[0])    # 1 < uncertainty score < inf            
        uncertainty_score=torch.Tensor(uncertainty_score)
    return uncertainty_score


def get_clusters(loader):
    
    model = models.resnet18(pretrained=True)
    #model = torch.hub.load('pytorch/vision', 'resnet18', weights='ResNet18_Weights.IMAGENET1K_V1')
    #model = torch.hub.load('pytorch/vision', 'resnet18')
    model = torch.nn.Sequential(*list(model.children())[:-1])

    features = []

    with torch.no_grad():
        for images, labels, filenames in loader:
            output = model(images)
            features.append(output.squeeze().numpy())
    features = np.concatenate(features)

    kmeans = KMeans(n_clusters=NUM_CLASSES, random_state=0).fit(features)

    labels = kmeans.labels_

    return labels

################### Main ###################
if __name__ == '__main__':
    
    y = [];
    x = [i*9 for i in range(CYCLES)]
    counts = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    num_samples = []
    
    addendum_test = ADDENDUM
    
    for trial in range(TRIALS): ## TRIALS=1
        indices = list(range(NUM_TRAIN)) # in config.py, we defined NUM_TRAIN = 10000

        ##### With no initial set #####
        labeled_set=[]
        first_selection_labeled_set=[]

        unlabeled_set = list(set(indices).difference(set(labeled_set)))   # with no labelled set, this is just list(set(indices))
        random_querylist=torch.randperm(NUM_TRAIN) # create a random index list instead of the query list.
        
        
        train_loader = DataLoader(data_train, 
                                    batch_size=BATCH, # batchsize of the train_loader is the num_classes
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    pin_memory=True)
        
        test_loader  = DataLoader(data_test, 
                                    batch_size=BATCH # config.py
                                    ) 
        
        # dict for dataloaders

        dataloaders  = {'train': train_loader, 'test': test_loader}
            
        ##### Load Pretrained FSL Model #####
        
        resnet18    = resnet.resnet18(num_classes=NUM_CLASSES).cuda()
        
        # load the weights from the pretrained model
        state_dict = torch.load('/home/jupyter-nschiavo@ualberta.-a5539/realcode/Active-FSL/Pretrain_FSL_Model/model_best_standard.pth.tar')     
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
            

        ##### Active learning cycles #####
        for cycle in range(CYCLES): # cycles in config

            criterion      = nn.CrossEntropyLoss(reduction='none')
            
            # stochastic gradient descent
            #optim_backbone = optim.SGD(models['backbone'].fc.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            
            # adam optimizer
            optim_backbone = optim.Adam(models['backbone'].fc.parameters(), lr=LR, weight_decay=WDECAY)
            
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            
            # formatting for each cycle
            print('--------------------------------Cycle %d/%d--------------------------------' % (cycle, CYCLES-1))
            
            # Training
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL)
            
            # testing
            acc_all, acc = test(models, dataloaders, mode='test')
            
            # results of the testing
            print('Trial %d/%d || Cycle %d/%d || Label set size %d: Test acc %.3f%%' % (trial+1, TRIALS, cycle, CYCLES-1, len(labeled_set), acc_all))
            for i in range(NUM_CLASSES):
                print("Class %d Acc: %.3f" % (i,acc[i]))
                
            num_samples.append(len(labeled_set))
            
            # for plotting
            y.append(acc_all)
            
            # the training and test portion of the loop is now complete
            # what follows is the active learning portion
    
     
            # Create unlabeled dataloader for the unlabeled unlabeled_set
           
            unlabeled_loader = DataLoader(data_unlabeled, 
                                            batch_size=BATCH, # In config.py, BATCH=128
                                            sampler=SubsetSequentialSampler(unlabeled_set), 
                                            pin_memory=True)
           
            ##### initial unlabeled dataloader with NUM_TRAIN images in it. For TSNE PLOT. #####
            initial_unlabeled_loader=DataLoader(data_unlabeled, 
                                          batch_size=BATCH, 
                                          sampler=SubsetSequentialSampler(indices), 
                                          pin_memory=True)
            
            ##### Measure uncertainty of each sample in the unlabeled_set #####
            
            """ margin only """
            #uncertainty = get_uncertainty_margin(models, unlabeled_loader)
            #strategy = 'margin'
            
            
            """ entropy only """
            # uncertainty = get_uncertainty_entropy(models, unlabeled_loader, unlabeled_set)
            # strategy = 'entropy'
            
            """ margin and entropy unnormalized """ #best one so far
            uncertainty_marg = get_uncertainty_margin(models, unlabeled_loader)
            uncertainty_ent = get_uncertainty_entropy(models, unlabeled_loader, unlabeled_set)
            
            uncertainty = np.add(uncertainty_marg, uncertainty_ent)
            
            strategy = 'margin_entropy_unnormalized'
            
            """ margin and entropy normalized """
            
            #uncertainty_marg = get_uncertainty_margin(models, unlabeled_loader)
            #uncertainty_ent = get_uncertainty_entropy(models, unlabeled_loader, unlabeled_set)
            
            
            #uncertainty_marg_shifted = [x-1 for x in uncertainty_marg]
            
            #uncertainty = np.add(np.tanh(uncertainty_marg_shifted), np.tanh(uncertainty_ent))
            
            #strategy = 'margin_entropy_normalized'
            
            
            """ Test """
            #probabilities = probs(models, dataloaders, mode='test')
            #uncertainty = get_uncertainty_LC(models, unlabeled_loader)
            
            
            """ Arguments """
            arg = np.argsort(uncertainty)
            
            # optional randomizer for benchmarking
            # arg = arg[torch.randperm(len(arg))]
            # strategy = 'random'
            
            """ Plotting (for debugging) """
            
            #time_now = datetime.now().strftime("%m-%d-%H-%M")
            
            #plt.figure()
            #plt.scatter(range(len(uncertainty)), np.tanh(sorted(uncertainty_marg)), label='margin')
            #plt.scatter(range(len(uncertainty)), np.tanh(sorted(uncertainty_ent)), label='entropy')
            #plt.scatter(range(len(uncertainty)), sorted(uncertainty), label='overall')
            
            #plt.legend()
            #plt.savefig('img/' + time_now + '_' + strategy + '_' + str(cycle) + '.png')
            
            
            
            
            ##### First round selection - individually selected samples #####
            
            num_splits = NUM_CLASSES
            
            splits = np.array_split(arg, ADDENDUM) #try reversed(arg)
            
            revised_arg = splits[0:num_splits] # try num_splits
            
            # ADDENDUM most informative samples method
            # first_selection_K_samples=list(torch.tensor(unlabeled_set)[arg][-ADDENDUM:].numpy()) 
            
            # even selection method
            #step = len(unlabeled_set)//NUM_CLASSES + 1
            #first_selection_K_samples=list(torch.tensor(unlabeled_set)[arg][::step].numpy())
            
            # moving even selection method
            first_selection_K_samples = []
            loc = 0
            
            for split in revised_arg:
                loc = cycle #cycle * len(split)//CYCLES
                first_selection_K_samples.append(split[loc])
                
            ##### Second round selection - pseudocomplete sets #####
            
            list0=[]
            second_selection_samples=[]
            
            #clusters = get_clusters(unlabeled_loader)
            
            """ Regular pseudocomplete sets"""
            
            '''
            for item in torch.flip(arg,[0]): # arg is the query list (indices)
                #true_label=data_unlabeled.targets[item] # target is the true label
                p_label = pseudo_labels[item]
                #cluster_label = clusters[item]
                if list0.count(p_label)<NUM_SHOTS:
                    list0.append(p_label)
                    second_selection_samples.append(item)
                if len(list0)>(NUM_CLASSES*NUM_SHOTS-1): # sample a complete N-way one-shot support set with pseudolabels
                    break
            '''

            
            
            """ evenly selected pseudo complete sets """
            
            """
            splits = np.array_split(arg, ADDENDUM)
            i = 0
            while (len(list0) < NUM_CLASSES and i < 10):
                for splitnum, phase in enumerate(reversed(splits)):
                    for item in torch.flip(phase,[0]):             # arg is the query list                         
                        p_label = pseudo_labels[item]               # using pseudo labels
                        #p_label = data_unlabeled.targets[item]     # using true labels
                        #if list0.count(p_label) == 0 and (counts[p_label, splitnum] == 0 or np.all(counts):
                        if list0.count(p_label) == 0 and splitnum < 6:
                            second_selection_samples.append(item)
                            list0.append(p_label)
                            counts[p_label, splitnum] += 1
                            break
                i += 1
            """    
            
            
            ##### Print Statistics #####
            
            ##### First selection statistics #####
            first_selection_K_samples_labels=[]
            for i in first_selection_K_samples:
                first_selection_K_samples_labels.append(data_train.targets[i])
            first_selection_K_samples_distribution=Counter(first_selection_K_samples_labels)
            
            selected_p_labels=[]
            # pseudo labels are pre generated labels from another algorithm (basically feature mapping)
            #for sample_index in first_selection_K_samples:
            #    selected_p_labels.append(pseudo_labels[sample_index])
            #print("First selection indices:",first_selection_K_samples)
            #print("First selection pseudo labels:  ",selected_p_labels)
            print("First selection true labels:    ",first_selection_K_samples_labels)
            print("First selection distribution:   ",first_selection_K_samples_distribution)
            
            
            ##### Second selection statistics #####
            
            second_selection_samples_labels=[]
            second_selection_samples_p_labels = []
            for i in second_selection_samples:
                second_selection_samples_labels.append(data_train.targets[i])
                second_selection_samples_p_labels.append(clusters[i])
            second_selection_samples_distribution=Counter(second_selection_samples_labels)
            second_selection_samples_p_distribution=Counter(second_selection_samples_p_labels)
            #print("Row: true label, Col: # times picked from bin")
            #print(counts)
            #print("Second selection true labels:    ",second_selection_samples_labels)
            #print("Second selection pseudo labels:  ",second_selection_samples_p_labels)
            #print("Query pseudo label distribution: ", second_selection_samples_p_distribution)
            #print("Query true label distribution:   ",second_selection_samples_distribution)
            
            
            ##### Update the labeled dataset and the unlabeled dataset, respectively #####
            
            # use one of the below depending on the selection
            
            labeled_set += first_selection_K_samples              
            #labeled_set += second_selection_samples
            
            # gathering the true labels for the queried data
            labeled_set_labels=[]
            for i in labeled_set:
                labeled_set_labels.append(data_train.targets[i])
            
            
            
            labeled_set_distribution=Counter(labeled_set_labels)
            print("The entire labeled set distribution: ",labeled_set_distribution)
            
            
            
            ##### Update the unlabeled set #####
            unlabeled_set = list(set(indices).difference(set(labeled_set))) 
            
            ##### Update the train_loader #####
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
            
    
    
    ##### Relevant data for analysis #####
    
    # plt.figure()
    # plt.scatter(x, y)
    # plt.xlabel("Labelled Examples")
    # plt.ylabel("Test Accuracy")
    # plt.title("Adam Optimizer, LR = {}, Seed {}".format(LR, r_seed))
    # plt.savefig('img/acc_graph.png')
    
    # np.set_printoptions(precision=3)
    # print(np.array(y))
    
    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    num_samples_total = num_samples[-1]
    
    np.savetxt('accuracies/' + time_now + '_' + strategy + '_seed' + str(RANDOM_SEED) + '_evenwbigshift.csv', np.c_[num_samples, y], fmt=['%d', '%.3f'], header='Labelled Samples, Accuracy', delimiter=',')