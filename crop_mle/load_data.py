import rasterio as rio
import geopandas as gpd
from _types import CropTypeDictionary
from dataclasses import asdict

# Load Model Predictions (GeoTIFF) and Extract Relevant Bands
def load_raster(raster_path):
    with rio.open(raster_path) as src:
        band3 = src.read(3)  # Crop Predictions
        band4 = src.read(4)  # Confidence Scores
        transform = src.transform
    return src, band3, band4, transform

def read_raster_to_array(raster_path):
    with rio.open(raster_path) as src:
        data = src.read()
        nodata = src.nodata
    return data, nodata

def read_raster_to_dataset(raster_path):
    with rio.open(raster_path) as src:
        return src

# Load Ground Truth Data (GPKG)
def load_gt(gt_path):
    return gpd.read_file(gt_path)

def schema_check(fields:gpd.GeoDataFrame, label_map:CropTypeDictionary):
    label_map_dict = asdict(label_map)
    crop_dict = label_map_dict['crop_dict']
    _fields = fields.copy() # copy so we can use original to look at non-conforming fields for info message
    _fields["normalized_label"] = fields["normalized_label"].apply(lambda x: next((k for k, v in crop_dict.items() if x in v), None))
    non_conforming = _fields[_fields["normalized_label"].isna()]
    if non_conforming.shape[0] > 0:
        print(f"Fields with non-conforming crop types: {non_conforming.shape[0]}")
        print(f"Non-conforming fields: {non_conforming['field_id'].values}\n {fields.loc[non_conforming.index].normalized_label}")
        keep = _fields.dropna(subset=['normalized_label'])
        conforming = fields.loc[keep.index]
    else:
        conforming = fields
    return conforming
