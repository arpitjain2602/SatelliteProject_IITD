import os
import glob
import pandas as pd
import numpy as np
from shutil import copyfile
from shutil import move

# Run remove duplicate before this

direc_1 = '/scratch/cse/mtech/mcs172873/data_l7_12band/bf/1'
direc_2 = '/scratch/cse/mtech/mcs172873/data_l7_12band/bf/2'
direc_3 = '/scratch/cse/mtech/mcs172873/data_l7_12band/bf/3'
direc_null = '/scratch/cse/mtech/mcs172873/data_l7_12band/bf/null'
train_path = '/scratch/cse/mtech/mcs172873/data_l7_12band/bf/train'
test_path = '/scratch/cse/mtech/mcs172873/data_l7_12band/bf/test'

'''
image_path="/home/ictd/Desktop/Arpit/Satellite project/data/L7_Avg10-12"
dirs1=os.listdir(image_path)
files1=[]
for direc1 in dirs1:
       file1=glob.glob(os.path.join(image_path,direc1))
       files1.extend(file1)
files=[]
files=files1
'''

image_pathOrissa="/scratch/cse/mtech/mcs172873/tifSingle/croppedImagesOddisa"
image_pathMaha="/scratch/cse/mtech/mcs172873/tifSingle/croppedImagesMaha"
image_pathKer="/scratch/cse/mtech/mcs172873/tifSingle/croppedImagesKerela"
image_pathKar="/scratch/cse/mtech/mcs172873/tifSingle/croppedImagesKarnataka"
image_pathGuj="/scratch/cse/mtech/mcs172873/tifSingle/croppedImagesGuj"
image_pathBihar="/scratch/cse/mtech/mcs172873/tifSingle/croppedImagesBihar"
dirs1=os.listdir(image_pathOrissa)
dirs2=os.listdir(image_pathMaha)
dirs3=os.listdir(image_pathKer)
dirs4=os.listdir(image_pathKar)
dirs5=os.listdir(image_pathGuj)
dirs6=os.listdir(image_pathBihar)
files1=[]
files2=[]
files3=[]
files4=[]
files5=[]
files6=[]
for direc1 in dirs1:
       file1=glob.glob(os.path.join(image_pathOrissa,direc1))
       files1.extend(file1)
for direc2 in dirs2:
       file2=glob.glob(os.path.join(image_pathMaha,direc2))
       files2.extend(file2)
for direc3 in dirs3:
       file3=glob.glob(os.path.join(image_pathKer,direc3))
       files3.extend(file3)
for direc4 in dirs4:
       file4=glob.glob(os.path.join(image_pathKar,direc4))
       files4.extend(file4)
for direc5 in dirs5:
       file5=glob.glob(os.path.join(image_pathGuj,direc5))
       files5.extend(file5)
for direc6 in dirs6:
       file6=glob.glob(os.path.join(image_pathBihar,direc6))
       files6.extend(file6)
files=[]
files=files1+files2+files3+files4+files5+files6

# Label Path
df=pd.read_csv('/home/cse/mtech/mcs172873/Labels/VillageLabels_BF.csv')
village_code=df["Town/Village"].values
emp_label=df["Village_HHD_Cluster_BF"].values
actual_labels= [ int(c) for c in emp_label]
s1 = pd.Series(actual_labels,index=list(village_code))
s2 = s1.groupby(s1.index).first()


wotif = map(lambda x: x.split('.tif')[0], files)
list_dict = []
for e in wotif:
    (a,b) = e[:-6], e[-6:]
    list_dict.append((a,b))

new_file_paths = []
for a,b in list_dict:
    try:
        label = s2[int(b)]
        x = a + b + '@label_' + str(label) #+ '.tif'
        new_file_paths.append(x)
    except KeyError:
        x = a + b + '@label_' + 'NULL' #+ '.tif'
        new_file_paths.append(x) 

ll = map(lambda x: x.split('@label_')[1], new_file_paths)
anp = np.array(ll)
print(np.unique(anp,return_counts=True))

assets_1 = []
assets_2 = []
assets_3 = []
assets_null = []
for e in new_file_paths:
    if(str(e.split('@label_')[1]) == '1'):
        assets_1.append(e)
    elif(str(e.split('@label_')[1]) == '2'):
        assets_2.append(e)
    elif(str(e.split('@label_')[1]) == '3'):
        assets_3.append(e)
    else:
        assets_null.append(e)

assets_1_tif = map(lambda x:x.split('@label_')[0]+'.tif', assets_1)
assets_2_tif = map(lambda x:x.split('@label_')[0]+'.tif', assets_2)
assets_3_tif = map(lambda x:x.split('@label_')[0]+'.tif', assets_3)
assets_null_tif = map(lambda x:x.split('@label_')[0]+'.tif', assets_null)

#Checkassets_null
print('Check - Total Files ',len(assets_null) + len(assets_1)+ len(assets_2)+ len(assets_3))

print('Entering Transfer Loop')
assets_1_path = []
for e in assets_1_tif:
    a,b = os.path.split(e)
    x = os.path.join(direc_1,b)
    assets_1_path.append(x)
for i in range(len(assets_1_tif)):
    #print(i)
    copyfile(assets_1_tif[i], assets_1_path[i])
print('Done for 1')
assets_2_path = []
for e in assets_2_tif:
    a,b = os.path.split(e)
    x = os.path.join(direc_2,b)
    assets_2_path.append(x)
for i in range(len(assets_2_tif)):
    #print(i)
    copyfile(assets_2_tif[i], assets_2_path[i])
print('Done for 2')
assets_3_path = []
for e in assets_3_tif:
    a,b = os.path.split(e)
    x = os.path.join(direc_3,b)
    assets_3_path.append(x)
for i in range(len(assets_3_tif)):
    #print(i)
    copyfile(assets_3_tif[i], assets_3_path[i])
print('Done for 3')
assets_null_path = []
for e in assets_null_tif:
    a,b = os.path.split(e)
    x = os.path.join(direc_null,b)
    assets_null_path.append(x)
for i in range(len(assets_null_tif)):
    #print(i)
    copyfile(assets_null_tif[i], assets_null_path[i])
print('Done for Null')

print('Broken into 1,2,3 and null folder, now transferring into test and train')

path1 = direc_1
path2 = direc_2
path3 = direc_3
paths1 = os.listdir(path1)
paths2 = os.listdir(path2)
paths3 = os.listdir(path3)
lp1 =[]
for direc1 in paths1:
    x=glob.glob(os.path.join(path1,direc1))
    lp1.extend(x)
lp2 =[]
for direc1 in paths2:
    x=glob.glob(os.path.join(path2,direc1))
    lp2.extend(x)
lp3 =[]
for direc1 in paths3:
    x=glob.glob(os.path.join(path3,direc1))
    lp3.extend(x)
print('No. of Class 1 sample:',len(lp1))
print('No. of Class 2 sample:',len(lp2))
print('No. of Class 3 sample:',len(lp3))
print('Total Samples:', len(lp1) + len(lp2) + len(lp3))

import random

n1 = len(lp1)
n2 = len(lp2)
n3 = len(lp3)

ir1 = np.arange(n1)
ir1 = np.asarray(ir1,dtype=np.int32)
random.shuffle(ir1)
tl1 = int(0.8*n1)
tf1 = ir1[:tl1]  #Trainging indexes
test1 = ir1[tl1:]   #Test Indexes

ir2 = np.arange(n2)
ir2 = np.asarray(ir2,dtype=np.int32)
random.shuffle(ir2)
tl2 = int(0.8*n2)
tf2 = ir2[:tl2]  #Training Indexes
test2 = ir2[tl2:]    #Test Indexes

ir3 = np.arange(n3)
ir3 = np.asarray(ir3,dtype=np.int32)
random.shuffle(ir3)
tl3 = int(0.8*n3)
tf3 = ir3[:tl3]     #Training Indexes
test3 = ir3[tl3:]     #Test Indexes


print('For Class 1:')
print('Training Files are:', len(tf1))
print('Testing Files are:', len(test1))
print('---------------------------------')
print('For Class 2:')
print('Training Files are:', len(tf2))
print('Testing Files are:', len(test2))
print('---------------------------------')
print('For Class 3:')
print('Training Files are:', len(tf3))
print('Testing Files are:', len(test3))


tfs1 = []
for i in tf1:
    x = lp1[i]
    tfs1.append(x)

tfs2 = []
for i in tf2:
    x = lp2[i]
    tfs2.append(x)

tfs3 = []
for i in tf3:
    x = lp3[i]
    tfs3.append(x)
    

    
testf1 = []
for i in test1:
    x = lp1[i]
    testf1.append(x)   

testf2 = []
for i in test2:
    x = lp2[i]
    testf2.append(x)   
    
testf3 = []
for i in test3:
    x = lp3[i]
    testf3.append(x)


print('Total Training Files:', len(tfs1) + len(tfs2) + len(tfs3))
print('Total Test Files:', len(testf1) + len(testf2) + len(testf3))
print('Total Files:',len(tfs1) + len(tfs2) + len(tfs3) + len(testf1) + len(testf2) + len(testf3))



for e in tfs1:
    a,b = os.path.split(e)
    x = os.path.join(train_path,b)
    move(e,x)

print('Done for 1')
    
for e in tfs2:
    a,b = os.path.split(e)
    x = os.path.join(train_path,b)
    move(e,x)
    
print('Done for 2')
    
for e in tfs3:
    a,b = os.path.split(e)
    x = os.path.join(train_path,b)
    move(e,x)

print('Done for 3')
    
print('Train files moved')
print('Total train files:', len(os.listdir(train_path)) ) 



for e in testf1:
    a,b = os.path.split(e)
    x = os.path.join(test_path,b)
    move(e,x)

for e in testf2:
    a,b = os.path.split(e)
    x = os.path.join(test_path,b)
    move(e,x)
    
for e in testf3:
    a,b = os.path.split(e)
    x = os.path.join(test_path,b)
    move(e,x)

print('Test Files Copied')
print('Total Test Files:', len(os.listdir(test_path)) )