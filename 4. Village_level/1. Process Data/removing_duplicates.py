import os
import glob
from shutil import copyfile

image_pathOrissa="/home/ictd/Desktop/Arpit/Satellite project/data/Landsat7_2010_2011_2012_Avg"

dirs1=os.listdir(image_pathOrissa)
files1=[]
for direc1 in dirs1:
       file1=glob.glob(os.path.join(image_pathOrissa,direc1))
       files1.extend(file1)
files=[]
files=files1

print('Total Files:',len(files))
files_code = map(lambda x: x.split('@')[3].split('.')[0].split()[0] , files)
uniques = list(set(files_code))
print('Unique Files ',len(uniques))

wotif = map(lambda x: x.split('.tif')[0], files)
list_dict = []
for e in wotif:
    (a,b) = e[:-6], e[-6:]
    list_dict.append((a,b))

dup_list = []
final_vcs = []
i=1
for ele in list_dict:
    vc = ele[1]
    if (vc in dup_list):
        continue
    else:
        dup_list.append(vc)
        final_vcs.append(ele)
    i=i+1

file_paths = []
for a,b in final_vcs:
    x = a+b+'.tif'
    file_paths.append(x)

# Change the folder name accordingly
file_new = map(lambda x: x.replace('Landsat7_2010_2011_2012_Avg','L7_Avg10-12'), file_paths)

for i in range(185746):
    copyfile(file_paths[i], file_new[i])