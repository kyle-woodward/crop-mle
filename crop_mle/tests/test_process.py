import unittest
import geopandas as gpd
from crop_mle.process import process_field, aggregate_predictions


class TestProcess(unittest.TestCase):

    def setUp(self):
        # Set up example data
        self.fields = gpd.read_file("crop_mle/tests/test.gpkg")
        self.raster_path = "crop_mle/tests/test.tif"

    def test_process_field(self):
        field = self.fields.iloc[0]
        result = process_field(field, self.raster_path)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], field["field_id"])

    def test_aggregate_predictions(self):
        gdf = aggregate_predictions(self.raster_path, self.fields)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertEqual(len(gdf), len(self.fields))


if __name__ == "__main__":
    unittest.main()
