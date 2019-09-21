import tifffile as tf
import json
import sys
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle

np.seterr(divide='ignore', invalid='ignore')
# Path of folder where district-wise tiff files are stored
# input_path = r'D:\Satellite Project\District\02_Break_into_Districts\districtTiffFiles500'

data_resolution = '100'
input_path = join(r'D:\\', 'Projects', 'Satellite Project', 'District', '02_Break_into_Districts', 'districtTiffFiles' + data_resolution)
# input_path = join(r'.', 'Satellite Project', 'District', '02_Break_into_Districts', 'districtTiffFiles' + data_resolution)
total_bands = 13
total_bins = 10
# File path followed by name to store the pickle file for created bins
bins_pickle_path = join(r'D:\\', 'Projects', 'Satellite Project', 'District', '03_Binning_Feature_Creation')
# bins_pickle_path = join(r'.', 'Satellite Project', 'District', '03_Binning_Feature_Creation')
bins_pickle_name = 'bins_quantile_2011_pickle_' + data_resolution
bins_pickle_file = open(join(bins_pickle_path, bins_pickle_name), 'rb')
bins_pickle = pickle.load(bins_pickle_file)
# File path and name(with extension) to save the feature file created at the end
feature_file_path = join(r'D:\\', 'Projects', 'Satellite Project', 'District', '03_Binning_Feature_Creation')
# feature_file_path = join(r'.', 'Satellite Project', 'District', '03_Binning_Feature_Creation')
feature_file_name = '2011_districts_quant_'+data_resolution+'m_phaneesh.csv'
# File path of the labels file
label_file_path = join(r'D:\\', 'Projects', 'Satellite Project', 'District', '03_Binning_Feature_Creation', '2011_labels.csv')
# label_file_path = join(r'.', 'Satellite Project', 'District', '03_Binning_Feature_Creation', '2011_labels.csv')
# Final Data
final_data_name = r'data_2011_' + data_resolution + 'm_quant.csv'

dist_tiff_names = [f for f in listdir(input_path) if isfile(join(input_path, f))]
flattened_DataDictionary={}
printing_dictionary={}

for band_id in range(total_bands):
	dist_counter = 0
	for current_dist_file in dist_tiff_names:
		dist_counter += 1
		print(f'band: {band_id} dist#: {dist_counter}\t name: {current_dist_file}', end='\t')
		current_dist_file_path = join(input_path, current_dist_file)
		image = tf.imread(current_dist_file_path, key=0)
		imagenum=np.array(image)
		# break
		if(band_id<10):
			current_dist_data = np.array(image)[:,:, band_id]
		elif(band_id==10): #ndvi
			current_dist_data = np.array((image[:,:,3]-(image)[:,:,2])/((image)[:,:,3]+(image)[:,:,2]))
		elif(band_id==11): #ndbi
			current_dist_data = np.array((image[:,:,4]-(image)[:,:,3])/((image)[:,:,3]+(image)[:,:,4]))	
			# current_dist_data = (np.array(image)[:,:,4]-np.array(image)[:,:,3])/(np.array(image)[:,:,4]+np.array(image)[:,:,3])
		elif(band_id==12): #mndwi
			current_dist_data = np.array((image[:,:,1]-(image)[:,:,4])/((image)[:,:,1]+(image)[:,:,4]))
			# current_dist_data = (np.array(image)[:,:,1]-np.array(image)[:,:,4])/(np.array(image)[:,:,1]+np.array(image)[:,:,4])

		dist_data_flattened = current_dist_data.flatten()
		# Remove nan and zero values
		dist_data_flattened = dist_data_flattened[~np.isnan(dist_data_flattened)]
		dist_data_flattened = dist_data_flattened[dist_data_flattened != 0]
		
		# Replaces the previous value of the key with new one.
		flattened_DataDictionary[current_dist_file] = dist_data_flattened
		print('---Completed')
	# For visualizing the immage as a histogram
	# bin_list = [0,0.05,0.1,0.15,0.2,0.5,1] # replace with the actual bins for the band
	# plt.hist(dist_data_flattened,bins=bin_list)
	# plt.show()
	# print(band_id)

	if band_id == 9:
		continue
	bins = bins_pickle[band_id][0]

	print('band_id = ', band_id, ' bins = ', bins)
	for key, val in flattened_DataDictionary.items():
		tempArray = val.copy()
		binning = np.histogram(tempArray, bins=bins)
		str1 = key
		str2 = str1[:-5] # to remove .tif from file name
		distName_st_cen_cd_censuscode = str2.split('@')  
		if band_id == 0:
			currArray = np.array([int(distName_st_cen_cd_censuscode[2])])
			currArray = np.append(currArray,binning[0])
			printing_dictionary[distName_st_cen_cd_censuscode[2]] = currArray
		else:
			currArray = np.array(binning[0])
			printing_dictionary[distName_st_cen_cd_censuscode[2]] = np.append(printing_dictionary[distName_st_cen_cd_censuscode[2]],currArray)

output_col_list = ['dist_census_code']
col_temp = []
for band_id in range(total_bands):	
	if band_id == 9:
		# because we're not using band 9
		continue
	col_temp.extend(['b_'+str(band_id)+'_'+str(t) for t in range(total_bins)])

output_col_list.extend(col_temp)
district_df = pd.DataFrame.from_dict(printing_dictionary, orient='index', columns=output_col_list)
print(district_df.shape)
district_df = district_df.sort_values(['dist_census_code'])
district_df.to_csv(join(feature_file_path, feature_file_name), index=False)

# Append labels to create the dataset
labels_df = pd.read_csv(label_file_path)
output_df = pd.merge(district_df, labels_df, left_on='dist_census_code', right_on='District_2011')
# to remove the extra columns which has same values as dist_census_code
## TRY DROPDUPLICATES
output_df = output_df.drop(columns='District_2011')
output_df.to_csv(join(feature_file_path, final_data_name), index=False)

