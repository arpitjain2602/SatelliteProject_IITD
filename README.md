# SatelliteProject_IITD
This is the repo for SatelliteProject under Prof. Aaditeshwar Seth

The whole project is divided into 2 parts
1. District Level
2. Village Level

## Folder Structure

### Papers
Contains research papers for related work, papers that we sent to various conferences, plots & tables for those papers

### Ground_Truth
The socio-economic indicators that we used are
1. BF (Bathroom Facility)
2. CHH (Condition of Household)
3. FC (Fuel for Cooking)
4. MSL (Main Source of Light)
5. MSW (Main Source of Water)
6. Asset (Assets in the household - radio/TV etc.)

District Level - Cleaned data from 2001 and 2011 census of India
Contains both actual values and classification labels that we used in our analysis

Village Level - Have data from 2011 census only


### Village_level
Folder for village level analysis

1. Process Data:
	- Contains files for downloading village images from Google Earth Engine (to be included soon)
	- Breaking the state level files into village level files using village shape files (link for shapefiles to be provided soon)
	- Processing those village level files (removing duplicates and the files for which we don't have ground truth)
2. Cross-sectional:
	- Classification:
		- 12 Bands - codes for incorporating 12 bands in the ResNet50 model (Code in PyTorch)
		- 3 Bands - codes for analysis on 3 bands. Scripts from ICTD Lab PC to be included here
		- PCA - code written by me and Ashutosh to to first PCA reduce the images (12 bands) and then perform classification


### District_level
Folder for district level analysis

1. Results: (Trained models with parameters, scores, input_features and targets)
	- Contains results for cross-sectional classification (2001 and 2011), Change-Classifier, Temporal Transferability
2. download_preprocess: 
	- Download data using Google Earth Engine
	- Break statefiles into district files using district shapefiles
	- 