import unittest
import geopandas as gpd
from shapely.geometry import Polygon
from crop_mle.select_fields import conf_percentiles, select_records


class TestSelection(unittest.TestCase):

    def setUp(self):
        # Set up example data
        self.input_gdf = gpd.GeoDataFrame(
            {
                "field_id": [1, 2],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                    Polygon([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]),
                ],
                "gt_label": [1, 2],
                "pred_label": [1, 2],
                "confidence": [0.9, 0.8],
            }
        )

    def test_conf_percentiles(self):
        percentiles = conf_percentiles(self.input_gdf, "pred_label", "confidence")
        self.assertIsInstance(percentiles, dict)

    def test_select_records(self):
        sr = select_records(
            self.input_gdf,
            "pred_label",
            "gt_label",
            "confidence",
        )
        self.assertIsInstance(sr, gpd.GeoDataFrame)


if __name__ == "__main__":
    unittest.main()
