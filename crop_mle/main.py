from crop_mle.process import evaluate_model, plot_confusion_matrix, aggregate_predictions
from load_data import load_gt, schema_check
from dataclasses import asdict
from _types import CropTypeDictionary

def main():
    
    gt_path = "/home/kyle/Downloads/crop-mle/u0c_gt_filtered_2022.gpkg" 
    raster_path = "/home/kyle/Downloads/crop-mle/ml_2021-08-01_2022-12-31_u0c.tif"
    labels_dict = CropTypeDictionary()

    fields = load_gt(gt_path)  # Load your fields DataFrame here
    fields = schema_check(fields, CropTypeDictionary())

    # preds = pd.read_csv("/home/kyle/Downloads/crop-mle/preds.csv")
    preds = aggregate_predictions(raster_path, fields)
    print(preds)
   
    cm, f1_scores = evaluate_model(fields, preds, labels_dict)
    print("F1 Scores per Crop:", f1_scores)
    
    label_list = list(asdict(labels_dict)['crop_dict'].keys())
    plot_confusion_matrix(cm, 
                          label_list, 
                          output_path="/home/kyle/Downloads/crop-mle/cm.png")
    print('done')
if __name__ == "__main__":
    main()