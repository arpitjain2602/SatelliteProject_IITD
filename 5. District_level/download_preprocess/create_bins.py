# this is a separate file because creating bins take time and need to created only once per data dataset.

import pickle
import numpy as np
import tifffile as tf
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import KBinsDiscretizer

# Path of folder where district-wise tiff files are stored
# input_path='districtTiffFiles11'
# input_path=r'D:\Satellite Project\District\districtTiffFiles'
# input_path = r'D:\Satellite Project\Anuj_Arpit\27janwork\districtTiffFiles11'
data_resolution = '100'
input_path = join(r'D:','Projects', 'Satellite Project', 'District', '02_Break_into_Districts', 'districtTiffFiles' + data_resolution)
# input_path = join(r'.', ,'Projects', 'Satellite Project', 'District', '02_Break_into_Districts', 'districtTiffFiles' + data_resolution)

total_bands = 13
total_bins = 10
year_of_sat_image = 2011
binning_strategy = 'quantile' # 'kmeans' 'uniform'
# File path followed by name to store the pickle file for created bins
bins_pickle_path = join(r'D:', 'Projects', 'Satellite Project', 'District', '03_Binning_Feature_Creation', 'bin_pickles')
# bins_pickle_path = join(r'.', 'Projects', 'Satellite Project', 'District', '03_Binning_Feature_Creation', 'bin_pickles')
bins_pickle_name = 'bins_' + str(total_bins) + '_' + binning_strategy + '_' + str(year_of_sat_image) + '_pickle_' + data_resolution
# Create list names of district tiff files
dist_tiff_names = [f for f in listdir(input_path) if isfile(join(input_path, f))]

# For visualization
flattened_DataDictionary={}
# Unused array
alldim1=np.array([])

length = []
bins_pickling=[]
bands_list = [0,1,2,3,4,5,6,7,8,9,10,11,12]
for band_id in bands_list:
	counter = 0
	all_districts_one_band_flattened_array = None
	for current_dist_file in dist_tiff_names:
		counter += 1	
		print(f'band: {band_id} dist#: {counter} \t fname:{current_dist_file}', end=' ')
		current_dist_file_path = join(input_path, current_dist_file)
		# Get array of current distric file for all bands
		image = tf.imread(current_dist_file_path, key=0)
		
		# break
		if (band_id<10):
			current_dist_data = np.array(image[:,:,band_id])
		elif (band_id ==10):
			current_dist_data = np.array((image[:,:,3]-(image)[:,:,2])/((image)[:,:,3]+(image)[:,:,2]))
		elif (band_id ==11):		
			current_dist_data = np.array((image[:,:,4]-(image)[:,:,3])/((image)[:,:,3]+(image)[:,:,4]))	
		else:
			current_dist_data = np.array((image[:,:,1]-(image)[:,:,4])/((image)[:,:,1]+(image)[:,:,4]))

		dist_data_flattened = current_dist_data.flatten()
		np.append(alldim1, dist_data_flattened)
		# to remove all zero values because zeros have no meaning here
		dist_data_flattened = dist_data_flattened[dist_data_flattened != 0]

		flattened_DataDictionary[current_dist_file] = dist_data_flattened

		if (all_districts_one_band_flattened_array is None):
			all_districts_one_band_flattened_array = dist_data_flattened
			length = [len(dist_data_flattened)]
		else:
			all_districts_one_band_flattened_array = np.append(all_districts_one_band_flattened_array,dist_data_flattened)
			length.append(len(dist_data_flattened))
		print('---Completed')

	# Removal of nan values
	all_districts_one_band_flattened_array = all_districts_one_band_flattened_array[~np.isnan(all_districts_one_band_flattened_array)]
	# all_districts_one_band_flattened_array=np.nan_to_num(all_districts_one_band_flattened_array)

	all_districts_one_band_flattened_array = all_districts_one_band_flattened_array.reshape(-1,1)
	
	# Feature Creation by dividing continuous data into bins
	est = KBinsDiscretizer(n_bins=total_bins, encode='ordinal', strategy=binning_strategy)
	est.fit(all_districts_one_band_flattened_array)
	bins = est.bin_edges_
	print(f'band_id: {band_id}\nBins: {bins}')
	bins_pickling.append(bins)

	f = open(join(bins_pickle_path, bins_pickle_name +'_band_'+ str(band_id)), 'wb')
	pickle.dump(bins, f)
	f.close()

f = open(join(bins_pickle_path, bins_pickle_name), 'wb')
pickle.dump(bins_pickling, f)
f.close()
