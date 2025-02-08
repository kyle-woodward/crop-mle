from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LabelsDictionary:
    crop_types: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class CropTypeDictionary(LabelsDictionary):
    crop_dict: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "Grassland Cultivated": ["grassland_cultivated"],
            "Grassland Nature": ["grassland_nature"],
            "Clover": ["clover"],
            "Alfalfa": ["alfalfa"],
            "Ryegrass": ["ryegrass"],
            "Winter Barley": ["barley_winter"],
            "Spring Barley": ["barley_spring", "barley_summer"],
            "Winter Wheat": ["wheat_winter"],
            "Triticale": ["triticale_winter", "triticale_spring"],
            "Winter Rye": ["rye_winter"],
            "Spring Rye": ["rye_spring"],
            "Spring Wheat": ["wheat_spring"],
            "Rice": ["rice"],
            "Millet": ["millet"],
            "Sorghum": ["sorghum", "Sorghum"],
            "Spring Oats": ["oats_spring"],
            "Winter Oats": ["oats_winter"],
            "Sunflowers": ["sunflowers", "sunflower"],
            "Flax": ["flax"],
            "Canola": ["canola_spring", "canola_winter"],
            "Grain Corn": ["corn_grain"],
            "Silage Corn": ["corn_silage"],
            "Potatoes": ["potatoes"],
            "Sugarbeets": ["sugarbeets", "beets"],
            "Soybeans": ["soybeans"],
            "Peas": ["peas", "peas_winter"],
            "Beans": ["beans"],
            "Lentils": ["lentils"],
            "Fallow": ["fallow"],
            "Turnips": ["turnips", "turnip"],
            "Trees": ["trees", "orchard", "orchards", "bananas"],
            "Vineyard": ["vineyard"],
        }
    )
    crop_numeric: Dict[str, int] = field(
        default_factory=lambda: {
            "Grassland Cultivated": 0,
            "Grassland Nature": 1,
            "Clover": 2,
            "Alfalfa": 3,
            "Ryegrass": 4,
            "Winter Barley": 5,
            "Spring Barley": 6,
            "Winter Wheat": 7,
            "Triticale": 8,
            "Winter Rye": 9,
            "Spring Rye": 10,
            "Spring Wheat": 11,
            "Rice": 12,
            "Millet": 13,
            "Sorghum": 14,
            "Spring Oats": 15,
            "Winter Oats": 16,
            "Sunflowers": 17,
            "Flax": 18,
            "Canola": 19,
            "Grain Corn": 20,
            "Silage Corn": 21,
            "Potatoes": 22,
            "Sugarbeets": 23,
            "Soybeans": 24,
            "Peas": 25,
            "Beans": 26,
            "Lentils": 27,
            "Fallow": 28,
            "Turnips": 29,
            "Trees": 30,
            "Vineyard": 31,
        }
    )
