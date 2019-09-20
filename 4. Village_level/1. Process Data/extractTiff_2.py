from json import loads
import sys
import rasterio
from rasterio.mask import mask

tiffFileName=r"C:\Users\AJain7\Downloads\SatImages\bihar_2007\Bihar_2007-0000000000-0000018944.tif"
jsonFileName=r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\village shapefiles\xaa.json"
startIdStr='BR'
dirName=r"C:\Users\AJain7\Downloads\SatImages\All_Images_2007"

# States Code - BR GJ KR [in xaa]
# States Code - KR, MH KL [in xab]
# States Code - MH, OD [in xac]

print('read')
stateData = loads(open(jsonFileName).read())
print('done')

for currVillageFeature in stateData["features"]:
    try:
        vCode2011=currVillageFeature["properties"]["village_code_2011"]
        vCode2001=currVillageFeature["properties"]["village_code_2001"]
        vId=currVillageFeature["properties"]["ID"]
        #print(vId[:2])
        
        if (vId[:2] != startIdStr):
            #print('Yes')
            continue
        
        #filename = dirName+"@"+str(vCode2001)+"@"+vId+"@"+str(vCode2011)+".tif"
        #print(filename)
        geoms=currVillageFeature["geometry"]
        #print(geoms)
        listGeom=[]
        listGeom.append(geoms)
        geoms=listGeom
        #print(geoms)
        
        with rasterio.open(tiffFileName) as src:
            #print('Enter')
            out_image, out_transform = mask(src, geoms, crop=True)
            #print('Exit')
        
        #print('Enter2')
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff","height": out_image.shape[1],"width": out_image.shape[2],"transform": out_transform})
        suppport_str = "\\"+ tiffFileName.split('\\')[6].split('.')[0]
        filename = dirName+suppport_str+"@"+str(vCode2001)+"@"+vId+"@"+str(vCode2011)+".tif"
        #print(filename)
        with rasterio.open(filename, "w", **out_meta) as dest:
            dest.write(out_image)
    except:
        continue
