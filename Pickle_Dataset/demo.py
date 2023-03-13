import os, cv2
from pickled import *
from load_data import *

# data_path = './test'
# file_list = './test_list.txt'
data_path = './train'
file_list = './train_list.txt'
save_path = './bin'

if __name__ == '__main__':
  data, label, lst = read_data(file_list, data_path, shape=224) #shape = image size
  #pickled(save_path, data, label, lst, bin_num = 1,mode="test")
  pickled(save_path, data, label, lst, bin_num = 2 ,mode="train") 
#bin_num depends on the dataset scale.E.G.there are 50000 images in NCT training set, then bin_num can be 5.
#There are only 1500 images in Breakhis dataset, so we set bin_num=2.
