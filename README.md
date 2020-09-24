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

### District_level
Folder for district level analysis

1. Results: (Trained models with parameters, scores, input_features and targets) - Refer Reading Results.ipynb on how to read results
	- Contains results for cross-sectional classification (2001 and 2011), Change-Classifier, Temporal Transferability
	
	- ChangeClassifier: 
		- ASSET_change_CC_FormalEMP%LIT_XGB.pkl - Asset change classifier including formal employment, literacy using trained XGBoost model
		- ASSET_change_CC_FormalEMP&Lit&&Emp_XGB.pkl - Asset change classifier including formal employment, literacy & emp. using XGBoost model
		- ASSET_change_CC_FormalEMP&Lit&&Emp_XGB_WO_2001.pkl - Asset change classifier including formal employment, literacy, emp but not using 2001 labels trained using XGBoost model
		- ASSET_change_CC_FormalEMP&Lit_XGB_WO_2001.pkl - Asset change classifier including formal employment, literacy but not using 2001 labels trained using XGBoost model
		- ASSET_change_CC_Only_2003-2011.pkl - Asset change classifier using labels only from 2003 to 2011
	
	- Temporal_Transferability:
		- ASSET_2001_TT2011_base_XGB.pkl - predicting 2001 using remaining data (backward model). Can find more information on what were the inputs, outputs and parameters by reading the pickle file. 
		- ASSET_2001_TT_FEMP&LIT_2011_base_XGB.pkl - predicting 2001, used FEMP & LIT apart from other to predict
		- ASSET_2011_TT_base_XGB.pkl - predicting 2011 using remaining data (forward model)
		- ASSET_2011_TT_FEMP&LIT_base_XGB.pkl

	- xgBoost-Results-ArpitNewFeatures_V1 (folder containing results of models trained for 2001)
		- ASSET_2001AnujMethod__top_score_specs@10.pkl - predicting Asset_2001 using 12 bands data & breaking them into 10 quantiles (denoted by "@10.pkl" and similarly for others)
	- xgBoost-Results-ArpitNewFeatures_V1_2011 (folder containing results of models trained for 2011)
		- ASSET_2011AnujMethod2011__top_score_specs@10.pkl - predicting Asset_2011 using 12 bands data & breaking them into 10 quantiles (denoted by "@10.pkl" and similarly for others)

	- Results-ArpitNew_V2 - Ignore (old features)
	- xgBoost-Results-AnujFeatures_V1 - Ignore (old features)
	- xgBoost-Results-AnujFeatures_V2 - Ignore (old features)
	- xgBoost-Results-PhaneeshFeatures_V1 - Ignore (old features)

2. download_preprocess: 
	- Download data using Google Earth Engine
	- Break statefiles into district files using district shapefiles
	- Cleaning and removing redundant files
	- Creating features from district files using quantile binning
3. Cross_sectional:
	- Contains scripts for model training on 2011 and 2001 data; Predicting cross year
	- BinsData_AnujMethod/2001_Anuj_Method - Contains features for 2001 data for different quantiles. Used this to train models. Ignore others
	- BinsData_AnujMethod/2003_Anuj_Method - Contains features for 2003 data for different quantiles. Used this in training backward and forward models and change classifier
	- BinsData_AnujMethod/2011_Anuj_Method - Contains features for 2011 data for different quantiles. Used this to train models. Ignore others
	- BinsData_ArpitMethod - Ignore (was attempting to create features using a different method, but continued with using features created by Anuj)
	- Arpit's new Features - Ignore
	- Anuj's Features - Ignore
	- Features_Arpit_New_2001.ipynb - file used for training cross-section models (the results of this model training are stored in Results -> xgBoost-Results-ArpitNewFeatures_V1 )
	- Features_Arpit_New_2011.ipynb - file used for training cross-section models (the results of this model training are stored in Results -> xgBoost-Results-ArpitNewFeatures_V1_2011 )
	- Results-Regenerate_cross_sectional.ipynb : re-generate the results for cross-sectional using already trained models
	- Results-Regenerate results & predict for other years using best model.ipynb
	- Results_Future-Regnerate results for future years using best models.ipynb


4. Temporal_Transferability:
	- scripts for training model for backward and forward models (Results available in Results -> Temporal_Transferability)
	- model_training
		- input_data/ASSET_CC.csv: contains columns such as  - census_code, predictions_2001, predictions_2003, ...predictions_2011
		- input_data_2011/ASSET_CC.csv: contains columns such as  - census_code, predictions_2001, predictions_2003, ...predictions_2011
		- input_data_2011-19/ASSET_CC.csv: contains columns such as  - census_code, predictions_2011, predictions_2013, ...predictions_2019
		- Temporal_Transferability_2001.ipynb - predicting 2011
		- Temporal_Transferability_2011.ipynb - predicting 2001
		- Results-Regenerate results & predict for other years using best model.ipynb - also present in the folder change_classifier. Used to generate input_data, input_data_2011, input_data_2019 folders

5. Change classifier:
	- scripts for training model on predictions to predict for future/past
	- input_files - files used as input for change classifier model

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


