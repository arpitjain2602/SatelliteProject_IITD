from json import loads
import sys
import rasterio
from rasterio.mask import mask
import os
import glob
from shutil import copyfile
import pandas as pd
import numpy as np
from shutil import copyfile
from shutil import move

# To do
# 1. Break file from State level to Village level
# 2. Remove duplicates from Village level files
# 3. Break files from Village level to label wise and into test and train data

# Variable Paths
output_directory = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\PreProcessData\broken_files" # The path where village file break hokar aaegi from state level
# Go to line 181 or search file_new - Basically in Remove Duplicates
image_path=r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\PreProcessData\broken_files_woDupli" # The link to broken village files with duplicates removed

# Final Paths to images - Create these folders
direc_1 = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\PreProcessData\data\msw\1"
direc_2 = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\PreProcessData\data\msw\2"
direc_3 = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\PreProcessData\data\msw\3"
direc_null = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\PreProcessData\data\msw\null"
train_path = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\PreProcessData\data\msw\train"
test_path = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\PreProcessData\data\msw\test"


# One time Paths
folder_containing_tifffiles = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\PreProcessData\median_6_states"
# SHAPE FILES********************************************************
# States Code - BR GJ KR [in xaa]
# States Code - KR, MH KL [in xab]
# States Code - MH, OD [in xac]
xaa_file = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\village shapefiles\xaa.json"
xab_file = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\village shapefiles\xab.json"
xac_file = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\village shapefiles\xac.json"

# Label Path
df=pd.read_csv(r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\Labels\VillageLabels_MSW.csv")
village_code=df["Town/Village"].values
emp_label=df["Village_HHD_Cluster_MSW"].values
actual_labels= [ int(c) for c in emp_label]
s1 = pd.Series(actual_labels,index=list(village_code))
s2 = s1.groupby(s1.index).first()












def break_state_file(state_Id_String, village_shape_file_list, output_directory, tiff_file):
    for shape_file in village_shape_file_list:
        print ('State:', state_Id_String,'---','ShapeFile:', shape_file[-8:],'---','TiffFile:',tiff_file)
        stateData = loads(open(shape_file).read())
        for currVillageFeature in stateData["features"]:
            try:
                vCode2011=currVillageFeature["properties"]["village_code_2011"]
                vCode2001=currVillageFeature["properties"]["village_code_2001"]
                vId=currVillageFeature["properties"]["ID"]
                if (vId[:2] != state_Id_String):
                    continue
                geoms=currVillageFeature["geometry"]
                listGeom=[]
                listGeom.append(geoms)
                geoms=listGeom
                with rasterio.open(tiff_file) as src:
                    out_image, out_transform = mask(src, geoms, crop=True)
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff","height": out_image.shape[1],"width": out_image.shape[2],"transform": out_transform})
                suppport_str = "\\"+ tiff_file.split('\\')[6].split('.')[0]
                filename = output_directory+suppport_str+"@"+str(vCode2001)+"@"+vId+"@"+str(vCode2011)+".tif"
                with rasterio.open(filename, "w", **out_meta) as dest:
                    dest.write(out_image)
            except:
                continue

bihar_shape_files = [xaa_file]
gujrat_shape_files = [xaa_file]
karnataka_shape_files = [xaa_file, xab_file]
maha_shape_files = [xab_file, xac_file]
kerala_shape_files = [xab_file]
orissa_shape_files = [xac_file]
# ******************************************************************
# for tifffile in os.listdir(folder_containing_tifffiles):
#     tifffile_path = os.path.join(folder_containing_tifffiles,tifffile)
#     if (tifffile[:2] == 'Bi'):
#         #run code for bihar
#         break_state_file('BR', bihar_shape_files, output_directory, tifffile_path)
#     elif (tifffile[:2] == 'Ka'):
#         #run code for Karnataka
#         break_state_file('KR', karnataka_shape_files, output_directory, tifffile_path)
#     elif (tifffile[:2] == 'Ma'):
#         #run code for Maha
#         break_state_file('MH', maha_shape_files, output_directory, tifffile_path)
#     elif (tifffile[:2] == 'Ke'):
#         #run code for Kerala
#         break_state_file('KR', kerala_shape_files, output_directory, tifffile_path)
#     elif (tifffile[:2] == 'Gu'):
#         #run code for Gujrat
#         break_state_file('GJ', gujrat_shape_files, output_directory, tifffile_path)
#     elif (tifffile[:2] == 'Or'):
#         #run code for Gujrat
#         break_state_file('OD', orissa_shape_files, output_directory, tifffile_path)

























# ******************************************************************************************************************************************
# Removing Duplicates     ***********************************************

# dirs1=os.listdir(output_directory)
# files1=[]
# for direc1 in dirs1:
#        file1=glob.glob(os.path.join(output_directory,direc1))
#        files1.extend(file1)
# files=[]
# files=files1

# print('Total Files:',len(files))
# files_code = map(lambda x: x.split('@')[3].split('.')[0].split()[0] , files)
# uniques = list(set(files_code))
# print('Unique Files ',len(uniques))

# wotif = map(lambda x: x.split('.tif')[0], files)
# list_dict = []
# for e in wotif:
#     (a,b) = e[:-6], e[-6:]
#     list_dict.append((a,b))

# dup_list = []
# final_vcs = []
# i=1
# for ele in list_dict:
#     vc = ele[1]
#     if (vc in dup_list):
#         continue
#     else:
#         dup_list.append(vc)
#         final_vcs.append(ele)
#     i=i+1

# file_paths = []
# for a,b in final_vcs:
#     x = a+b+'.tif'
#     file_paths.append(x)

# # Change the folder name accordingly
# # CHANGE THE BELOW PEICE ACCORDING TO OUTPUT DIRECTORY KI LOCATION
# file_new = list(map(lambda x: x.replace('broken_files','broken_files_woDupli'), file_paths))

# for i in range(len(file_paths)):
#     copyfile(file_paths[i], file_new[i])



























# ******************************************************************************************************************************************
# Breaking into folders     ***********************************************

dirs1=os.listdir(image_path)
files1=[]
for direc1 in dirs1:
       file1=glob.glob(os.path.join(image_path,direc1))
       files1.extend(file1)
files=[]
files=files1

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

assets_1_tif = list(map(lambda x:x.split('@label_')[0]+'.tif', assets_1))
assets_2_tif = list(map(lambda x:x.split('@label_')[0]+'.tif', assets_2))
assets_3_tif = list(map(lambda x:x.split('@label_')[0]+'.tif', assets_3))
assets_null_tif = list(map(lambda x:x.split('@label_')[0]+'.tif', assets_null))

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

