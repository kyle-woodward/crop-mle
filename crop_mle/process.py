import numpy as np
import rasterio as rio
from rasterio.mask import mask
import multiprocessing as mp
import pandas as pd
import geopandas as gpd
import logging
import os
from datetime import datetime

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), "..", ".log")
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime("logfile_%Y%m%d_%H%M%S.log")
log_filepath = os.path.join(log_dir, log_filename)

logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def process_field(field: gpd.GeoDataFrame, raster_path: str) -> tuple:
    """
    Aggregates prediction raster at `raster_path` to the `field` geomtry, taking the majority class and mean confidence.

    Args:
        field (gpd.GeoDataFrame): GeoDataFrame containing field geometry.
        raster_path (str): Path to the prediction raster.

    Returns:
        tuple: Field ID, majority class, mean confidence
    """
    with rio.open(raster_path) as src:
        geom = [field.geometry]
        try:
            out_image, _ = mask(
                src, geom, crop=True, nodata=-99
            )  # account for edge pixels give them intentional nodata value
            croptype = out_image[2, :, :]
            croptype_nona = croptype[croptype != -99]  # remove edge pixels

            conf = out_image[3, :, :]
            conf_nona = conf[conf != -99]  # remove edge pixels
            if croptype_nona.size > 0:  # if we have valid pixels in the field
                flattened = croptype_nona.flatten()
                majority_class = np.bincount(flattened).argmax()
                avg_conf = np.mean(conf_nona)
            else:
                logging.info(f"No pixels in field {field.field_id}")
                majority_class, avg_conf = None, None
        except Exception as e:
            logging.info(
                f"Error processing field {field['field_id']}:\n Traceback: {e}"
            )
            logging.info(f"croptype: {croptype}")
            logging.info(f"conf: {conf}")
            majority_class, avg_conf = None, None

    return field["field_id"], majority_class, avg_conf


def aggregate_predictions(raster_path: str, fields: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Parallelizes the process_field function to aggregate predictions for each field in the fields GeoDataFrame.

    Args:
        raster_path (str): Path to the prediction raster.
        fields (gpd.GeoDataFrame): GeoDataFrame containing field geometries.

    Returns:
        pd.DataFrame: DataFrame containing field_id, predicted class, and confidence.
    """
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            process_field, [(field, raster_path) for _, field in fields.iterrows()]
        )
    return pd.DataFrame(results, columns=["field_id", "predicted_int", "confidence"])
