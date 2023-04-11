
import numpy as np 
import os

def label2num(s): # here you should make changes according to your dataset. turn label names to number,
        digits = {'lungaca': '0', 'lungscc': '1', 'lungn': '2', 'colonca': '3', 'colonn': '4'}
        return digits[s]
    
savePath="./train_list.txt"
picturePath = './train'   
# savePath="./test_list.txt"
# picturePath = './test' 

file_txt = open(savePath ,'w')  
names= os.listdir(picturePath)
res = ''
labelnum = 99999999

for fi in names:
    if fi.endswith('.jpeg'):
        #label.append(fi.split("-")[0])   #This line should be customed according to your image names. 
        res = ''.join([i for i in fi if not i.isdigit()])
        res = res.split('.')[0]
        labelnum = (label2num(res))
        file_txt.write(fi +" " + labelnum +'\n')  

file_txt.close() 