import logging, os
from datetime import datetime

# import geopandas as gpd

# Set up logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".log"))
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime("logfile_%Y%m%d_%H%M%S.log")
log_filepath = os.path.join(log_dir, log_filename)

logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def conf_percentiles(input_gdf, pred_column: str, confidence_column: str):
    """
    Calculate confidence percentiles for each predicted label.

    Args:
        input_gdf (gpd.GeoDataFrame): Input GeoDataFrame containing the data.
        pred_column (str): Column name for the predicted labels.
        confidence_column (str): Column name for the confidence values.

    Returns:
        dict: Dictionary with predicted labels as keys and DataFrames of percentiles as values.
    """
    percentiles = [0.5]  # Add more percentiles if needed
    conf_percentiles = {}
    for label in input_gdf[pred_column].unique():
        label_conf = input_gdf[input_gdf[pred_column] == label][confidence_column]
        conf_percentiles[label] = label_conf.quantile(percentiles).to_frame().T
    return conf_percentiles


def select_records(
    input_gdf,
    pred_column: str,
    gt_column: str,
    confidence_column: str,
):
    """
    Select records based on prediction agreement and confidence thresholds.

    Args:
        input_gdf (gpd.GeoDataFrame): Input GeoDataFrame containing the data.
        pred_column (str): Column name for the predicted labels.
        gt_column (str): Column name for the ground truth labels.
        confidence_column (str): Column name for the confidence values.
        output_path (str): Path to save the selected records to a .gpkg file.

    Returns:
        None (saves the selected records to a .gpkg file)
    """
    confidence_percentiles = conf_percentiles(input_gdf, pred_column, confidence_column)
    for label in confidence_percentiles:
        logging.info(
            f"Keeping Correct Predictions for label '{label}' under Confidence threshold: {float(confidence_percentiles[label].iloc[0, 0])}"
        )
    # Create a dictionary to map predicted labels to their confidence cutoffs
    conf_cutoff_dict = {
        label: float(confidence_percentiles[label].iloc[0, 0])
        for label in confidence_percentiles
    }

    # Vectorized operation to set confidence cutoffs
    input_gdf["conf_cutoff"] = input_gdf[pred_column].map(conf_cutoff_dict)

    # Vectorized operation to apply the selection rules
    conditions = (input_gdf[pred_column] != input_gdf[gt_column]) | (
        (input_gdf[pred_column] == input_gdf[gt_column])
        & (input_gdf[confidence_column] < input_gdf["conf_cutoff"])
    )
    input_gdf["keep"] = conditions

    # Filter the GeoDataFrame
    out_gdf = input_gdf[input_gdf["keep"]]
    return out_gdf
