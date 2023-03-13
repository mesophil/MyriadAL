
import numpy as np 
import os

def label2num(s): # here you should make changes according to your dataset. turn label names to number,
        digits = {'ADI': '0', 'BACK': '1', 'DEB': '2', 'LYM': '3', 'MUC': '4','MUS': '5','NORM': '6','STR': '7','TUM': '8'}
        return digits[s]
    
# savePath="./train_list.txt"
# picturePath = './train'   
savePath="./test_list.txt"
picturePath = './test' 
file_txt = open(savePath ,'w')  
names= os.listdir(picturePath)
i=0
label=[]
labelnum=[]
for fi in names: #
    #if fi.endswith(".png"):       
        label.append(fi.split("-")[0])   #This line should be customed according to your image names. 
        labelnum.append(label2num(label[i]))
        file_txt.write(fi +" " +labelnum[i]+'\n')  
        i+=1
file_txt.close() 