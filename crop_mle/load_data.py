import geopandas as gpd
from _types import CropTypeDictionary
from dataclasses import asdict
import logging, os
from datetime import datetime

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), '..', '.log')
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime('logfile_%Y%m%d_%H%M%S.log')
log_filepath = os.path.join(log_dir, log_filename)

logging.basicConfig(filename=log_filepath, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_gt(gt_path:str) -> gpd.GeoDataFrame:
    """"
    Load ground truth vector data from file path.
    
    Args:
        gt_path (str): Path to the ground truth vector data file.
    
    Returns:
        gpd.GeoDataFrame: Ground truth vector data.
    """
    return gpd.read_file(gt_path)

def schema_check(fields:gpd.GeoDataFrame,
                 label_col: str, 
                 label_map:CropTypeDictionary) -> gpd.GeoDataFrame:
    """
    Conducts a schema check of the fields data to ensure all crop type values in `label_col` have a matching label `label_map`. 
        Non-conforming records are removed from the GeoDataFrame and are logged.

    Args:
        fields (gpd.GeoDataFrame): DataFrame containing field data.
        label_col (str): Column name for crop type values.
        label_map (CropTypeDictionary): Dictionary mapping crop types to labels.
    
    Returns:
        gpd.GeoDataFrame: DataFrame with only fields that have crop types in the label_map.
    """
    label_map_dict = asdict(label_map)
    crop_dict = label_map_dict['crop_dict']
    _fields = fields.copy() # copy so we can use original to look at non-conforming fields for info message
    _fields[label_col] = fields[label_col].apply(lambda x: next((k for k, v in crop_dict.items() if x in v), None))
    non_conforming = _fields[_fields[label_col].isna()]
    if non_conforming.shape[0] > 0:
        logging.info(f"Fields with crop types not in label_map: {non_conforming.shape[0]}")
        logging.info(f"Non-conforming fields: {non_conforming['field_id'].values}\n {fields.loc[non_conforming.index][label_col]}")
        keep = _fields.dropna(subset=[label_col])
        conforming = fields.loc[keep.index]
    else:
        conforming = fields
    return conforming
