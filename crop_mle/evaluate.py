import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from dataclasses import asdict
from _types import CropTypeDictionary

# Compute Confusion Matrix & F1 Scores
def evaluate_model(fields:gpd.GeoDataFrame,
                    preds:gpd.GeoDataFrame, 
                    label_map:CropTypeDictionary):
    label_map_dict = asdict(label_map)
    crop_dict = label_map_dict['crop_dict']
    crop_numeric = label_map_dict['crop_numeric']
    
    fields["mapped_crop_type"] = fields["normalized_label"].apply(lambda x: next((k for k, v in crop_dict.items() if x in v), None))
    preds["mapped_predicted_label"] = preds["predicted_label"].apply(lambda x: next((k for k, v in crop_numeric.items() if x == v), None))
    field_labels = fields["mapped_crop_type"]
    preds_labels = preds["mapped_predicted_label"]
    
    cm = confusion_matrix(field_labels, preds_labels, labels=list(crop_dict.keys()))
    f1_scores = f1_score(field_labels, preds_labels, average=None)
    return cm, f1_scores

# Plot Confusion Matrix
def plot_confusion_matrix(cm:np.ndarray, 
                          labels:list,
                          output_path:str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.show()