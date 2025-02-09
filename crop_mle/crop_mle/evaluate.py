import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from dataclasses import asdict
from crop_mle._types import CropTypeDictionary
from collections import OrderedDict
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


def schema_check(
    fields: gpd.GeoDataFrame, label_col: str, label_map: CropTypeDictionary
) -> gpd.GeoDataFrame:
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
    crop_dict = label_map_dict["crop_dict"]
    _fields = (
        fields.copy()
    )  # copy so we can use original to look at non-conforming fields for info message
    _fields[label_col] = fields[label_col].apply(
        lambda x: next((k for k, v in crop_dict.items() if x in v), None)
    )
    non_conforming = _fields[_fields[label_col].isna()]
    if non_conforming.shape[0] > 0:
        logging.info(
            f"Fields with crop types not in label_map: {non_conforming.shape[0]}"
        )
        logging.info(
            f"Non-conforming fields: {non_conforming['field_id'].values}\n {fields.loc[non_conforming.index][label_col]}"
        )
        keep = _fields.dropna(subset=[label_col])
        conforming = fields.loc[keep.index]
    else:
        conforming = fields
    return conforming


def standardize_labels(
    df: gpd.GeoDataFrame, gt_label: str, pred_label: str, label_map: CropTypeDictionary
) -> gpd.GeoDataFrame:
    """ "
    Create Standardized "gt_label" and "pred_label" columns in the DataFrame
        by re-mapping values from non-standardized input gt_label and pred_label columns to standardized values from label_map.
    Args:
        df (gpd.GeoDataFrame): DataFrame containing ground truth and predicted label columns.
        gt_label (str): Column name for ground truth labels.
        pred_label (str): Column name for predicted labels.
        label_map (CropTypeDictionary): Dictionary mapping crop types to labels.

    Returns:
        gpd.GeoDataFrame: DataFrame with standardized "gt_label" and "pred_label" columns.
    """
    label_map_dict = asdict(label_map)
    crop_dict = label_map_dict["crop_dict"]
    crop_numeric = label_map_dict["crop_numeric"]

    # map gt labels pred ints to same standardized labels from label_map
    df["gt_label"] = df[gt_label].apply(
        lambda x: next((k for k, v in crop_dict.items() if x in v), None)
    )
    df["pred_label"] = df[pred_label].apply(
        lambda x: next((k for k, v in crop_numeric.items() if x == v), None)
    )
    return df


def cm_f1(
    gt_pred_df: gpd.GeoDataFrame,
    gt_label: str,
    pred_label: str,
    label_map: CropTypeDictionary,
) -> tuple:
    """
    Compute the confusion matrix and F1 scores for the given ground truth and predicted labels.

    Args:
        gt_pred_df (gpd.GeoDataFrame): DataFrame containing ground truth and predicted labels.
        gt_label (str): Column name for ground truth labels.
        pred_label (str): Column name for predicted labels.
        label_map (CropTypeDictionary): Dictionary mapping crop types to labels.

    Returns:
        tuple: Confusion matrix (np.ndarray) and F1 scores DataFrame (pd.DataFrame).
    """
    label_map_dict = asdict(label_map)
    crop_dict = label_map_dict["crop_dict"]

    # prep gt & pred labels for confusion matrix and f1-score
    gt_labels = gt_pred_df[gt_label]
    preds_labels = gt_pred_df[pred_label]

    # we want to account for and remove classes with zero instances in both gt and pred when reporting f1-score
    all_classes = list(crop_dict.keys())

    # Compute confusion matrix
    cm = confusion_matrix(gt_labels, preds_labels, labels=all_classes)

    # Identify classes with zero instances in both ground truth and predictions
    zero_instance_classes = [
        all_classes[i]
        for i in range(len(all_classes))
        if cm[i].sum() == 0 and cm[:, i].sum() == 0
    ]

    # Compute F1 scores
    f1_scores = f1_score(
        gt_labels, preds_labels, labels=all_classes, average=None, zero_division=0
    )

    # Create an OrderedDict to map classes to their F1 scores, preserving the order from label_map
    f1_scores_dict = OrderedDict(
        (cls, round(float(score), 2)) for cls, score in zip(all_classes, f1_scores)
    )

    # Remove classes with zero instances from the F1 scores dictionary
    for cls in zero_instance_classes:
        del f1_scores_dict[cls]

    # Convert OrderedDict to DataFrame with keys as the first column
    f1_scores_df = pd.DataFrame(list(f1_scores_dict.items()), columns=["Crop", "F1"])

    return cm, f1_scores_df


def record_count(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Count the occurrences of each unique value in the specified column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to count unique values.

    Returns:
        pd.DataFrame: DataFrame with counts of unique values.
    """
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "Count"]
    return counts


def agreement(df: pd.DataFrame, gt_label: str, pred_label: str) -> pd.DataFrame:
    """
    Calculate the percentage agreement between ground truth and predicted labels.

    Args:
        df (pd.DataFrame): DataFrame containing ground truth and predicted labels.
        gt_label (str): Column name for ground truth labels.
        pred_label (str): Column name for predicted labels.

    Returns:
        pd.DataFrame: DataFrame with a 'Percent Agreement' column aligned to gt_label column index.
    """
    total_counts = df[gt_label].value_counts()
    match_counts = df[df[gt_label] == df[pred_label]][gt_label].value_counts()

    # Fill missing values in match_counts with zeros
    match_counts = match_counts.reindex(total_counts.index, fill_value=0)

    percent_agreement = round((match_counts / total_counts * 100), 2).reset_index()
    percent_agreement.columns = [gt_label, "Percent Agreement"]
    return percent_agreement


def average_confidence(
    df: pd.DataFrame, pred_label: str, confidence_col: str
) -> pd.DataFrame:
    """
    Calculate the average confidence values for each unique predicted label.

    Args:
        df (pd.DataFrame): DataFrame containing predicted labels and confidence values.
        pred_label (str): Column name for predicted labels.
        confidence_col (str): Column name for confidence values.

    Returns:
        pd.DataFrame: DataFrame with average confidence values for each unique predicted label.
    """
    avg_conf_df = round(df.groupby(pred_label)[confidence_col].mean(), 2).reset_index()
    avg_conf_df.columns = [pred_label, "Average Confidence"]
    return avg_conf_df


def plot_confusion_matrix(cm: np.ndarray, labels: list, output_path: str):
    """
    Plot and save the confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        labels (list): List of labels for the confusion matrix.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(14, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        annot_kws={"size": 8},
    )
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()  # Adjust layout to make room for labels
    plt.savefig(output_path)
