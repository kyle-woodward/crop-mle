import unittest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import numpy as np
from crop_mle.evaluate import (
    schema_check,
    standardize_labels,
    record_count,
    agreement,
    average_confidence,
    cm_f1,
)
from crop_mle._types import CropTypeDictionary
from dataclasses import asdict


class TestProcess(unittest.TestCase):

    def setUp(self):
        # Set up example data
        self.fields = gpd.read_file("crop_mle/tests/test.gpkg")
        self.raster_path = "crop_mle/tests/test.tif"
        self.label_map = CropTypeDictionary()

    def test_schema_check(self):

        # Test schema_check function
        fields = self.fields
        result = schema_check(
            fields,
            "normalized_label",
            self.label_map,
        )
        self.assertIn(
            list(result.normalized_label.unique()),
            list(asdict(self.label_map)["crop_dict"].values()),
        )

    def test_standardize_labels(self):

        # Test standardize_labels function
        fields = self.input_gdf = gpd.GeoDataFrame(
            {
                "field_id": [1],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                    # Polygon([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]),
                ],
                "normalized_label": ["wheat_winter"],
                "predicted_int": [7],
                "confidence": [0.9],
            }
        )
        result = standardize_labels(
            fields,
            "normalized_label",
            "predicted_int",
            self.label_map,
        )
        crop_dict_keys = list(asdict(self.label_map)["crop_dict"].keys())
        for label in result.gt_label.unique():
            self.assertIn(label, crop_dict_keys)
        for label in result.pred_label.unique():
            self.assertIn(label, crop_dict_keys)
        self.assertIsInstance(result, gpd.GeoDataFrame)

    def test_record_count(self):

        # Test record_count function
        fields = self.fields
        result = record_count(
            fields,
            "normalized_label",
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_agreement(self):

        # Test agreement function
        self.df = pd.DataFrame(
            {
                "field_id": [1, 2, 3],
                "gt_label": ["wheat_winter", "wheat_winter", "wheat_winter"],
                "pred_label": ["wheat_winter", "wheat_winter", "wheat_winter"],
            }
        )
        result = agreement(
            self.df,
            "gt_label",
            "pred_label",
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_average_confidence(self):

        # Test average_confidence function
        self.df = pd.DataFrame(
            {
                "field_id": [1, 2, 3],
                "pred_label": ["wheat_winter", "wheat_winter", "wheat_winter"],
                "confidence": [0.9, 0.8, 0.7],
            }
        )
        result = average_confidence(
            self.df,
            "pred_label",
            "confidence",
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(result["Average Confidence"].iloc[0], float)

    def test_cm_f1(self):

        # Test cm_f1 function
        self.df = pd.DataFrame(
            {
                "field_id": [1, 2, 3],
                "gt_label": ["Winter Wheat", "Winter Wheat", "Winter Wheat"],
                "pred_label": ["Winter Wheat", "Winter Wheat", "Winter Wheat"],
            }
        )
        result = cm_f1(self.df, "gt_label", "pred_label", self.label_map)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
