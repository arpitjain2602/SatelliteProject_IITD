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
import matplotlib.pyplot as plt
tiffFileName='landsat7_india_500_2011-01-01_2011-12-31.tif'
jsonFileName='Census_2011/2011_Dist.geojson'


countryData = json.loads(open(jsonFileName).read())
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
inputFolder='districtTiffFiles'
onlyfiles = [f for f in listdir(inputFolder) if isfile(join(inputFolder, f))]
flattened_DataDictionary={}



alldim1=np.array([])

printing_dictionary={}

for i in range(10):
	allFlattenedArray = None
	for currDFile in onlyfiles:
		#currDistrictFile='districtTiffFiles/Rajkot@24@476.tif'
		currDistrictFile='districtTiffFiles/'+currDFile
		tif = TIFF.open(currDistrictFile, mode='r')
		image = tif.read_image()
		
		# break

		dataAll = np.array(image)[:,:,i]
		# print(dataAll.shape)

		flattenData=dataAll.flatten()
		np.append(alldim1,flattenData)
		flattenData=flattenData[flattenData != 0]
		flattened_DataDictionary[currDFile]=flattenData

		if (allFlattenedArray is None):
			allFlattenedArray = flattenData        
		else:
			allFlattenedArray = np.append(allFlattenedArray,flattenData)
	allFlattenedArray=np.nan_to_num(allFlattenedArray)
	# sns.set_style('darkgrid')
	# sns.distplot(allFlattenedArray)

	# bin = [0,0.05,0.1,0.15,0.2,0.5,1]
	# bins_1 = np.array(bin)
	# plt.hist(allFlattenedArray,bins=100)
	# plt.show()
	print(i)
	for key, val in flattened_DataDictionary.items():
		tempArray=val.copy()
		# tempArray[tempArray>1]=1
		if (i==5 or i==6):		
			bins_6 = np.array([0,285,295,300,305,310,400])
			binning=np.histogram(tempArray, bins=bins_6)
		else:
			bins_1 = np.array([0,0.05,0.09,0.11,0.13,0.18,1])
			binning=np.histogram(tempArray, bins=bins_1)
		str1=key
		str2=str1[:-4]
		distName_st_cen_cd_censuscode=str2.split('@')  
		if i==0:
			currArray=np.array([int(distName_st_cen_cd_censuscode[2])])
			currArray=np.append(currArray,binning[0])
			printing_dictionary[distName_st_cen_cd_censuscode[2]]=currArray
		else:
			currArray=np.array(binning[0])
			printing_dictionary[distName_st_cen_cd_censuscode[2]]=np.append(printing_dictionary[distName_st_cen_cd_censuscode[2]],currArray)
	   
columns1=['censuscode']
col_help=['b_'+str(int(t/6)+1)+str(t%6+1) for t in range(60)]
columns1.extend(col_help)
dataframe_districts=pd.DataFrame.from_dict(printing_dictionary, orient='index',columns=columns1)
dataframe_districts.to_csv('2011_districts2.csv')