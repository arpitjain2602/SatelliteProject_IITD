from json import loads
import sys
import rasterio
from rasterio.mask import mask
import os

output_directory = r"C:\Users\AJain7\Downloads\SatImages\Maha_11Bands_Landsat8_VillageLevel"
folder_containing_tifffiles = r"C:\Users\AJain7\Downloads\SatImages\Maha_11Bands_Landsat8"

# SHAPE FILES********************************************************
# States Code - BR GJ KR [in xaa]
# States Code - KR, MH KL [in xab]
# States Code - MH, OD [in xac]
xaa_file = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\village shapefiles\xaa.json"
xab_file = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\village shapefiles\xab.json"
xac_file = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\village shapefiles\xac.json"

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



for tifffile in os.listdir(folder_containing_tifffiles):
    tifffile_path = os.path.join(folder_containing_tifffiles,tifffile)
    if (tifffile[:2] == 'Bi'):
        #run code for bihar
        break_state_file('BR', bihar_shape_files, output_directory, tifffile_path)
    elif (tifffile[:2] == 'Ka'):
        #run code for Karnataka
        break_state_file('KR', karnataka_shape_files, output_directory, tifffile_path)
    elif (tifffile[:2] == 'Ma'):
        #run code for Maha
        break_state_file('MH', maha_shape_files, output_directory, tifffile_path)
    elif (tifffile[:2] == 'Ke'):
        #run code for Kerala
        break_state_file('KR', kerala_shape_files, output_directory, tifffile_path)
    elif (tifffile[:2] == 'Gu'):
        #run code for Gujrat
        break_state_file('GJ', gujrat_shape_files, output_directory, tifffile_path)
    elif (tifffile[:2] == 'Or'):
        #run code for Gujrat
        break_state_file('OD', orissa_shape_files, output_directory, tifffile_path)
