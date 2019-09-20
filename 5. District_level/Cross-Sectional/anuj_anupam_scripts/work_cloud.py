# find /scratch/cse/mtech/mcs182021/images-2011 -type f -name "Bihar*" -print | xargs -d "\n" cp -t /scratch/cse/btech/cs1150214/images-2011/Bihar


import csv
import json
import sys
import pandas as pd
import rasterio
# from rasterio.tools.mask import mask
from rasterio.mask import mask
from libtiff import TIFF
import libtiff
libtiff.libtiff_ctypes.suppress_warnings()
import sys
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import pickle
import h5py
import scipy.misc
import seaborn as sns
import math
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
tiffFileName='landsat7_india_500_2011-01-01_2011-12-31.tif'
jsonFileName='Census_2011/2011_Dist.geojson'
np.seterr(divide='ignore', invalid='ignore')

# countryData = json.loads(open(jsonFileName).read())
# for currDistrictFeature in countryData["features"]:
#     # currDistrictFeature=countryData["features"][0]
#     distName=currDistrictFeature["properties"]['DISTRICT']
#     st_cen_cd=currDistrictFeature["properties"]['ST_CEN_CD']
#     censuscode=currDistrictFeature["properties"]['censuscode']
#     geoms=currDistrictFeature["geometry"]
#     listGeom=[]
#     listGeom.append(geoms)
#     geoms=listGeom
#     with rasterio.open(tiffFileName) as src:
#       out_image, out_transform = mask(src, geoms, crop=True)

#     out_meta = src.meta.copy()

#         # save the resulting raster  
#     out_meta.update({"driver": "GTiff",
#         "height": out_image.shape[1],
#         "width": out_image.shape[2],
#         "transform": out_transform})

#     with rasterio.open('districtTiffFiles/'+distName+'@'+str(st_cen_cd)+'@'+str(censuscode)+".tif", "w", **out_meta) as dest:
#       dest.write(out_image)
inputFolder='districtTiffFiles01/'
onlyfiles = [f for f in listdir(inputFolder) if isfile(join(inputFolder, f))]
flattened_DataDictionary={}


alldim1=np.array([])

printing_dictionary={}

with open('cloud_01_new.csv','w') as f:
	writer = csv.writer(f)
	writer.writerow(['District_code','cloud_01'])	

	for currDFile in onlyfiles:
		#currDistrictFile='districtTiffFiles/Rajkot@24@476.tif'
		currDistrictFile='districtTiffFiles01/'+currDFile
		tif = TIFF.open(currDistrictFile, mode='r')
		image = tif.read_image()
		imagenum=np.array(image)
		# break	
		dataAll = np.array(image)[:,:,9]
		
		# print(dataAll.shape)

		flattenData=dataAll.flatten()
		flattenData=flattenData[~numpy.isnan(flattenData)]
		# np.append(alldim1,flattenData)
		flattenData=flattenData[flattenData != 0]
		flattened_DataDictionary[currDFile]=flattenData

		j = 0
		for i in range(len(flattenData)):
			a = format(int(flattenData[i]),'011b')
			j+=int(a[4])
		# print(j)
		str1=currDFile
		str2=str1[:-4]
		distName_st_cen_cd_censuscode=str2.split('@')  

		if(len(flattenData)==0):
			writer.writerow([int(distName_st_cen_cd_censuscode[2]),0.0])
			# print(int(distName_st_cen_cd_censuscode[2]),0)
		else:
			writer.writerow([int(distName_st_cen_cd_censuscode[2]),float(j*100.0/len(flattenData))])
			# print(int(distName_st_cen_cd_censuscode[2]),float(j*100.0/len(flattenData)))
# sns.set_style('darkgrid')
# sns.distplot(allFlattenedArray)

# bin = [0,0.05,0.1,0.15,0.2,0.5,1]
# bins_1 = np.array(bin)
# plt.hist(allFlattenedArray,bins=100)
# plt.show()
# for key, val in flattened_DataDictionary.items():
# 	tempArray=val.copy()
# 	# tempArray[tempArray>1]=1
# 	if (i==0):
# 		bins = np.array([-0.00434329,  0.1007754 ,  0.11160686,  0.11821117,  0.1227451 , 0.12707032,  0.13192225,  0.13770965,  0.14507869,  0.15936665, 0.95032692])
# 		binning=np.histogram(tempArray, bins=bins)
# 	elif (i==1):
# 		bins = np.array([-0.01408871,  0.08253167,  0.0961787 ,  0.10382023,  0.1093633 , 0.11472073,  0.12038616,  0.12717851,  0.13725099,  0.16736212, 1.0420866 ])
# 		binning=np.histogram(tempArray, bins=bins)
# 	elif (i==2):
# 		bins = np.array([-0.00482686,  0.06614591,  0.08589753,  0.09778892,  0.10643996, 0.11453842,  0.12282554,  0.13290162,  0.14881481,  0.198098  , 1.02079499])
# 		binning=np.histogram(tempArray, bins=bins)
# 	elif (i==3):
# 		bins = np.array([-0.00755567,  0.18234085,  0.19880261,  0.20834468,  0.2161618 , 0.2238521 ,  0.23253816,  0.24297239,  0.25751856,  0.28109035, 1.27039528])
# 		binning=np.histogram(tempArray, bins=bins)
# 	elif (i==4):
# 		bins = np.array([-0.00433501,  0.12489212,  0.15201586,  0.16859764,  0.18297477, 0.19705071,  0.21264757,  0.23184341,  0.25823477,  0.30344069, 0.79090953])
# 		binning=np.histogram(tempArray, bins=bins)
# 	elif (i==5):
# 		bins = np.array([139.37446594, 290.237854  , 295.99209595, 297.51446533, 298.76654053, 300.00958252, 300.99514771, 302.6942749 , 304.61553955, 306.98297119, 316.6730957 ])
# 		binning=np.histogram(tempArray, bins=bins)
# 	elif (i==6):
# 		bins = np.array([240.0700531 , 290.18469238, 295.84790039, 297.39749146, 298.65072632, 299.88671875, 300.98419189, 302.60568237, 304.47180176, 306.95477295, 316.37011719])
# 		binning=np.histogram(tempArray, bins=bins)
# 	elif (i==7):
# 		bins = np.array([-0.01089788,  0.05854745,  0.08025672,  0.0967405 ,  0.10981411, 0.12208197,  0.13525279,  0.150858  ,  0.17411724,  0.21964493, 0.68281025])
# 		binning=np.histogram(tempArray, bins=bins)
# 	elif (i==8):
# 		bins = np.array([-0.01455807,  0.13366747,  0.14486524,  0.15203518,  0.15789008, 0.16345453,  0.16944915,  0.17672215,  0.18833737,  0.22320986, 1.14053929])
# 		binning=np.histogram(tempArray, bins=bins)
# 	elif (i==9):
# 		# bins = np.array([139.37446594, 295.99209595, 298.76654053, 300.99514771, 304.61553955, 316.6730957 ])
# 		# binning=np.histogram(tempArray, bins=bins)
# 		continue
# 	elif (i==10):
# 		bins = np.array([-1.12757325,  0.10881562,  0.19139672,  0.23961319,  0.27516243, 0.30721408,  0.3401936 ,  0.37869947,  0.43171915,  0.52279216, 1.04447377])
# 		binning=np.histogram(tempArray, bins=bins)
# 	elif (i==11):
# 		bins = np.array([-1.03576207, -0.26961386, -0.17666008, -0.12369751, -0.08304815, -0.04621166, -0.01102346,  0.02409801,  0.05986707,  0.09498096, 1.06299067])
# 		binning=np.histogram(tempArray, bins=bins)
# 	else:
# 		bins = np.array([-1.05333078, -0.38361807, -0.35001816, -0.32411161, -0.30181184, -0.28061783, -0.25744784, -0.22931152, -0.19161635, -0.12188197, 1.07649541])
# 		binning=np.histogram(tempArray, bins=bins)

# 	str1=key
# 	str2=str1[:-4]
# 	distName_st_cen_cd_censuscode=str2.split('@')  
# 	if i==0:
# 		currArray=np.array([int(distName_st_cen_cd_censuscode[2])])
# 		currArray=np.append(currArray,binning[0])
# 		printing_dictionary[distName_st_cen_cd_censuscode[2]]=currArray
# 	else:
# 		currArray=np.array(binning[0])
# 		printing_dictionary[distName_st_cen_cd_censuscode[2]]=np.append(printing_dictionary[distName_st_cen_cd_censuscode[2]],currArray)
	   
# columns1=['censuscode']
# col_help=['b_'+str(int(t/6)+1)+str(t%6+1) for t in range(78)]
# columns1.extend(col_help)
# dataframe_districts=pd.DataFrame.from_dict(printing_dictionary, orient='index')#,columns=columns1)
# dataframe_districts.to_csv('2011_cloud.csv')