var india_image_landsat = /* color: #d63000 */ee.Geometry.Polygon(
       [[[72.67721647172027, 33.70687327738794],
         [71.52829856946812, 28.376811894145625],
         [70.05613060071812, 28.096121839556172],
         [69.42469723366798, 27.064934355722027],
         [68.05699737015357, 24.52662941297628],
         [68.03502471390357, 23.07922068891511],
         [69.7024656745624, 20.952481593174337],
         [72.13933416440102, 19.077462878233206],
         [72.73259588315102, 16.714890866301445],
         [74.14468427838358, 14.014456179692006],
         [75.41909834088358, 7.716874371058136],
         [78.27554365338358, 7.237587883728653],
         [80.38491865338358, 10.234279305036768],
         [80.94966713681242, 15.045759029638015],
         [87.80513588681242, 20.730581600880377],
         [90.96919838681242, 21.222973998354885],
         [92.02388588681242, 19.61687214694604],
         [98.90981325558778, 28.439833441015523],
         [97.06411013058778, 29.936013104946785],
         [94.03188356808778, 29.554463466799483],
         [91.79067263058778, 28.32384430844447],
         [89.81313356808778, 28.40118449488407],
         [87.48403200558778, 28.207728480788063],
         [84.75942263058778, 29.209831968395275],
         [82.47426638058778, 30.278161861866334],
         [79.88149294308778, 31.746969850186243],
         [80.84828981808778, 35.47867577852967],
         [79.17836794308778, 36.29752782995585],
         [77.50844606808778, 36.15572767624159],
         [74.91567263058778, 37.45752609324931],
         [71.97133669308778, 36.36833144469512]]]);
          



var currState=india_image_landsat;


var yearStr='2019'
var year_trimester=1


var dateStart=(yearStr+'-01-01')
var dateEnd=(yearStr+'-12-31')

if (year_trimester==1)
{
 dateStart=(yearStr+'-01-01')
 dateEnd=(yearStr+'-04-17')
}
else if (year_trimester==2)
{
 dateStart=(yearStr+'-05-01')
 dateEnd=(yearStr+'-08-30')
}
else if (year_trimester==3)
{
 dateStart=(yearStr+'-09-01')
 dateEnd=(yearStr+'-12-31')
}


var bands = ee.List(['B1','B2','B3','B4','B5','B6_VCID_1','B6_VCID_2','B7','B8','BQA']);
var india_image = (ee.Image)(ee.ImageCollection('LANDSAT/LE07/C01/T1_TOA').select(bands)
.filterBounds(currState).filterDate(dateStart,dateEnd).filter(ee.Filter.lt('CLOUD_COVER',4)).median());

var bqaFloat=india_image.select('BQA').float();
print(bqaFloat);
var bands_nobqa = ee.List(['B1','B2','B3','B4','B5','B6_VCID_1','B6_VCID_2','B7','B8']);
india_image=india_image.select(bands_nobqa);
india_image=india_image.addBands(bqaFloat.select('BQA'))

// india_image_bqa=india_image.select('BQA')
// print (india)

// india_image=india_image.addBands(nl2013.select("constant"));

print(india_image);

Map.addLayer(india_image.clip(currState), {bands: ['B3', 'B2', 'B1'], max: 0.4}, '3bands');
// Map.addLayer(india_image.clip(currState), {bands: ['constant']}, 'night_lights');
Map.addLayer(india_image, {}, 'all');

Export.image.toDrive({
 image: india_image.clip(currState),
 description: 'landsat7_india_500_'+dateStart+'_'+dateEnd,
 scale: 500,
 folder:'ayushGEE',
 maxPixels: 10e9,
 region: currState
});