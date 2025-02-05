import numpy as np
import rasterio as rio
from rasterio.mask import mask
from multiprocessing import shared_memory, Pool
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import box
from sklearn.metrics import confusion_matrix, f1_score
from dataclasses import asdict
from _types import CropTypeDictionary
import geopandas as gpd

def process_field(field:gpd.GeoDataFrame, 
                  raster_path:str):
    with rio.open(raster_path) as src:
        geom = [field.geometry]
        try:
            out_image, _ = mask(src, geom, crop=True)
            croptype = out_image[2,:,:]  
            conf = out_image[3,:,:]  
            if out_image.size > 0: # if we have valid pixels in the field
                    flattened = croptype.flatten()
                    majority_class = np.bincount(flattened).argmax()
                    avg_conf = np.mean(conf)
            else:
                majority_class, avg_conf = None, None
        except Exception as e :
            print(f"Error processing field {field['field_id']}:\n Traceback: {e}")
            print(f"croptype: {croptype}")
            print(f"conf: {conf}")
            majority_class, avg_conf = None, None
    
    return field["field_id"], majority_class, avg_conf

def aggregate_predictions(raster_path:str, 
                          fields:gpd.GeoDataFrame):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(process_field, [(field, raster_path) for _, field in fields.iterrows()])
    return pd.DataFrame(results, columns=["field_id","predicted_label", "confidence"])

