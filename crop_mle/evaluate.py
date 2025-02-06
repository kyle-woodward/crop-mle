import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from dataclasses import asdict
from _types import CropTypeDictionary
from collections import OrderedDict

def standardize_labels(df: gpd.GeoDataFrame,
                        gt_label:str,
                        pred_label:str, 
                        label_map:CropTypeDictionary) -> gpd.GeoDataFrame:
    label_map_dict = asdict(label_map)
    crop_dict = label_map_dict['crop_dict']
    crop_numeric = label_map_dict['crop_numeric']
    
    # map gt labels pred ints to same standardized labels from label_map
    df["gt_label"] = df[gt_label].apply(lambda x: next((k for k, v in crop_dict.items() if x in v), None)) 
    df["pred_label"] = df[pred_label].apply(lambda x: next((k for k, v in crop_numeric.items() if x == v), None))
    return df

# Compute Confusion Matrix & F1 Scores
def cm_f1(gt_pred_df:gpd.GeoDataFrame,
                    gt_label:str,
                    pred_label:str, 
                    label_map:CropTypeDictionary) -> tuple:
    label_map_dict = asdict(label_map)
    crop_dict = label_map_dict['crop_dict']
    
    # prep gt & pred labels for confusion matrix and f1-score
    gt_labels = gt_pred_df[gt_label]
    preds_labels = gt_pred_df[pred_label]
    
    # we want to account for and remove classes with zero instances in both gt and pred when reporting f1-score
    all_classes = list(crop_dict.keys())
    
    # Compute confusion matrix
    cm = confusion_matrix(gt_labels, preds_labels, labels=all_classes)
    
    # Identify classes with zero instances in both ground truth and predictions
    zero_instance_classes = [all_classes[i] for i in range(len(all_classes)) if cm[i].sum() == 0 and cm[:, i].sum() == 0]
    
    # Compute F1 scores
    f1_scores = f1_score(gt_labels, 
                         preds_labels, 
                         labels=all_classes, 
                         average=None, 
                         zero_division=0)
    
    # Create an OrderedDict to map classes to their F1 scores, preserving the order from label_map
    f1_scores_dict = OrderedDict((cls, round(float(score),2)) for cls, score in zip(all_classes, f1_scores))

    # Remove classes with zero instances from the F1 scores dictionary
    for cls in zero_instance_classes:
        del f1_scores_dict[cls]
    
    # Convert OrderedDict to DataFrame with keys as the first column
    f1_scores_df = pd.DataFrame(list(f1_scores_dict.items()), columns=['Crop', 'F1'])
    
    return cm, f1_scores_df

def record_count(df: pd.DataFrame, 
                 column: str) -> pd.DataFrame:
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'Count']
    return counts

def agreement(df: pd.DataFrame, 
              gt_label: str, 
              pred_label: str) -> pd.DataFrame:
    total_counts = df[gt_label].value_counts()
    match_counts = df[df[gt_label] == df[pred_label]][gt_label].value_counts()
    
    # Fill missing values in match_counts with zeros
    match_counts = match_counts.reindex(total_counts.index, fill_value=0)
    
    percent_agreement = round((match_counts / total_counts * 100),2).reset_index()
    percent_agreement.columns = [gt_label, 'Percent Agreement']
    return percent_agreement

# Average Confidence Values
def average_confidence(df: pd.DataFrame, 
                       pred_label: str, 
                       confidence_col: str) -> pd.DataFrame:
    avg_conf_df = round(df.groupby(pred_label)[confidence_col].mean(),2).reset_index()
    avg_conf_df.columns = [pred_label, 'Average Confidence']
    return avg_conf_df

# Plot Confusion Matrix
def plot_confusion_matrix(cm:np.ndarray, 
                          labels:list,
                          output_path:str):
    plt.figure(figsize=(14, 14))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", annot_kws={"size": 8})
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()  # Adjust layout to make room for labels
    plt.savefig(output_path)
