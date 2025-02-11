from crop_mle.process import aggregate_predictions
from crop_mle.evaluate import (
    schema_check,
    standardize_labels,
    record_count,
    agreement,
    average_confidence,
    cm_f1,
    plot_confusion_matrix,
)
from crop_mle.select_fields import select_records
from dataclasses import asdict
from crop_mle._types import CropTypeDictionary
import pandas as pd
import geopandas as gpd
import time
import logging, os
from datetime import datetime
import argparse

# Set up logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".log"))
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime("logfile_%Y%m%d_%H%M%S.log")
log_filepath = os.path.join(log_dir, log_filename)

logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    """
    Main function to evaluate crop model predictions.

    Example usage:
        python main.py --gt /path/to/ground_truth.gpkg --raster /path/to/prediction_raster.tif
    """

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Crop Model Performance Analysis.",
        epilog="Example usage: python main.py --gt /path/to/ground_truth.gpkg --raster /path/to/prediction_raster.tif --label_field normalized_label --mode analysis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--gt", type=str, required=True, help="Path to the ground truth vector file"
    )
    parser.add_argument(
        "--raster",
        type=str,
        required=True,
        help="Path to the prediction raster data file",
    )
    parser.add_argument(
        "--label_field",
        type=str,
        default="normalized_label",
        help="Field name in the ground truth data containing crop type labels",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="analysis",
        choices=["select", "analysis"],
        help="choose 'select' for selecting underperforming fields or 'analysis' for evaluating model performance",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Directory to save the output files",
        required=False,
    )
    args = parser.parse_args()

    # load data, data dictionary, and conduct schema check on fields
    gt_path = args.gt
    raster_path = args.raster
    label_field = args.label_field
    mode = args.mode
    out_dir = args.out_dir

    print(f"Starting {args.mode} Process...check progress at {log_filepath}")
    if not args.out_dir:
        # we'll store results in a repo-level directory
        out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(out_dir, exist_ok=True)

    # initialize cropt tpye dictionary and access labels
    labels_dict = CropTypeDictionary()
    label_list = list(asdict(labels_dict)["crop_dict"].keys())

    # load ground truth data and conduct schema check
    fields = gpd.read_file(gt_path)
    fields = schema_check(fields, label_field, labels_dict)

    # aggregate model predictions to fields
    preds = aggregate_predictions(raster_path, fields)

    # merge ground truth and model predictions dataframes and save out to csv (useful for debugging)
    merged_df = fields.merge(
        preds, on="field_id", how="left", suffixes=("_field", "_pred")
    )
    merged_df.dropna(
        inplace=True
    )  # drop fields with no predictions (i.e. no pixels in field)

    merged_df = standardize_labels(merged_df, label_field, "predicted_int", labels_dict)
    if mode == "select":
        # select records based on confidence percentiles
        merged_df = select_records(merged_df, "pred_label", "gt_label", "confidence")
        merged_df.to_file(os.path.join(out_dir, "selected_fields.gpkg"), driver="GPKG")

    elif mode == "analysis":

        # compute counts by crop type and agreement between ground truth and model predictions
        counts_df = record_count(merged_df, "gt_label")
        agreement_df = agreement(merged_df, "gt_label", "pred_label")

        # get average confidence by crop type for model predictions
        avg_conf_df = average_confidence(merged_df, "pred_label", "confidence")

        # compute confusion matrix and F1 scores
        cm, f1_scores_df = cm_f1(merged_df, "gt_label", "pred_label", labels_dict)
        # Save confusion matrix csv/png
        plot_confusion_matrix(
            cm, label_list, output_path=out_dir + "/confusion_matrix.png"
        )
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"), index=True)

        # Merge crop counts, agreement and f1_scores for final .csv
        ct_agree = counts_df.merge(agreement_df, on="gt_label", how="left")
        f1_conf = f1_scores_df.merge(
            avg_conf_df, left_on="Crop", right_on="pred_label", how="left"
        )
        final_df = f1_conf.merge(
            ct_agree, left_on="Crop", right_on="gt_label", how="left"
        )

        # drop any crop types with no gt_label (i.e. no instances in the ground truth)
        final_df.dropna(subset=["gt_label"], inplace=True)

        # clean-up duplicate columns and write out final result
        final_df.drop(columns=["gt_label", "pred_label"], inplace=True)
        final_df.to_csv(os.path.join(out_dir, "final_results.csv"), index=False)
    else:
        raise ValueError("Invalid mode. Please select 'select' or 'analysis'.")

    logging.info("Done")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution Time: {(end_time - start_time)/60} mins")
