# Crop Doctor :man_health_worker:
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Key Features:

:corn: Performs crop model performance analysis

:mechanical_arm: Provides automated pipeline tooling for extracting underperforming fields

## Setup & Use

1. Build Docker image from Dockerfile:

`docker build -t crop-mle .`

2. Run the docker container:

`docker run -v /path/to/your/data:/data -e PYTHONPATH=/app -it crop-mle bash`

**ensure that `path/to/your/data` directory contains the two datasets that you need to download: [ml_2021-08-01_2022-12-31_u0c.tif](https://drive.google.com/file/d/1R_4NtTIUrQHo7cGA-Xi26KvUh3RYjLa3/view?usp=drive_link) and [u0c_gt_filtered_2022.gpkg](https://drive.google.com/file/d/1uOM9DlyNp4V2dNtW_cSt8VLvTRSzlGn6/view?usp=drive_link)**

3. From here you can:

* run in `analysis` mode to conduct performance analysis:

`python crop_mle/main.py --gt /data/u0c_gt_filtered_2022.gpkg --raster /data/ml_2021-08-01_2022-12-31_u0c.tif --label_field normalized_label --mode analysis`

* run in `select` mode to select export underperforming fields from `u0c` using our underperformance ruleset:

`python crop_mle/main.py --gt /data/u0c_gt_filtered_2022.gpkg --raster /data/ml_2021-08-01_2022-12-31_u0c.tif --label_field normalized_label --mode analysis`

## Implementation Notes

This tool utilizes `multiprocessing` and vectorized `pandas` operations for efficient raster to vector field-level aggregations and analysis of large tabular datasets. We also use `dataclass` as a clean way to store and load the provided (and any other future hypothetical) label dictionary. As a qualitative efficiency benchmark, a machine with 20 cores & 64GB RAM runs both processing tools in under 2 minutes each for the ~150k record fields dataset.

### Some Gotcha's
At some point, I discovered that two fields in the ground truth dataset had 'normalized_label':'radish' which wasn't in provided data dictionary. So I implemented a [`schema_check`](/crop_mle/load_data.py) to check for non-conforming field values and removing those records at the outset.

I also discovered through some [rasterio.mask.mask](https://rasterio.readthedocs.io/en/stable/api/rasterio.mask.html) debugging that there are about ~170 fields that are so small that the default behavior was not even grabbing one valid pixel. Those are removed from analysis by setting a nodata value for the masking and removing those nodata values before doing the field-level prediction aggregation.

## Diagnosis :stethoscope:

### Lowest Performers

Looking by F1 scores and Agreement % first, we see that there are 6 crop types in ground truth with 0 F1 and 0% agreement: Winter Rye, Spring Rye, Millet, Beans, Turnips, and Trees. 

| Crop       | F1 | Average Confidence | Count | Percent Agreement |
|------------|----|--------------------|-------|-------------------|
| Winter Rye | 0  |                    | 193   | 0                 |
| Spring Rye | 0  |                    | 17    | 0                 |
| Millet     | 0  | 35.78              | 9     | 0                 |
| Beans      | 0  |                    | 336   | 0                 |
| Turnips    | 0  |                    | 11    | 0                 |
| Trees      | 0  |                    | 909   | 0                 |


These are crop types that the model made either never predicted or predicted but was wrong in all instances (e.g. Millet). In Millet's case, it incorrectly predicted 2 fields each of Silage Corn, Grain Corn, and Fallow. While these are minority classes by a long shot - representing 0.96% of ground truth dataset (1,475 fields)- it is worth considering grabbing all of these fields for next training round as they do not imply much from a training overhead stand-point and nobody loves to see a 0.0 F1 score or Agreement.

### Middle of the Pack

Next up are crop types with an F1 above 0.0 and below 0.5:

| Crop           | F1   | Average Confidence | Count | Percent Agreement |
|----------------|------|--------------------|-------|-------------------|
| Spring Wheat   | 0.01 | 32.46              | 298   | 0.34              |
| Soybeans       | 0.07 | 54.01              | 33    | 66.67             |
| Vineyard       | 0.17 | 43.17              | 34    | 23.53             |
| Sorghum        | 0.18 | 42.4               | 105   | 14.29             |
| Triticale      | 0.19 | 47.79              | 311   | 45.98             |
| Clover         | 0.34 | 43.31              | 387   | 40.57             |
| Spring Oats    | 0.35 | 42.91              | 1131  | 30.59             |
| Winter Oats    | 0.39 | 47.11              | 184   | 40.22             |
| Spring Barley  | 0.44 | 46.85              | 4190  | 41.98             |
| Lentils        | 0.46 | 52.94              | 112   | 52.68             |
| Ryegrass       | 0.47 | 41.85              | 1885  | 53.05             |


In total, this subset of ground truth data represents about 5.6% (8670) of the total ground truth dataset. A curious blip here is Soybeans with an F-1 of 0.07 but a Percent agreement of 66.7 which I've diagnosed as a bad case of recall conflation: there were only 33 soybean fields in `u0c` ground truth dataset and the model got 22 of them - while also making over 500 incorrect predictions :sweat_smile:.

These above are also minority classes and in the same vein as the above group, it is certainly worth re-training with all of these as well from the same reasoning as the above group: any gains in improvement you get from retraining should be worth the little cost from a training overhead perspective. Interpreting 'what's good enough' is more to do with knowing the business use case, so I am making a broad assumption here that F1's below 0.5 are of concern regardless of model confidence in these predictions.

### Top Performers
Here's the top half (F1 > 0.5) of the performance dataset.

| Crop           | F1   | Average Confidence | Count | Percent Agreement |
|----------------|------|--------------------|-------|-------------------|
| Sugarbeets     | 0.96 | 77.03              | 11024 | 96.43             |
| Canola         | 0.96 | 78.29              | 10154 | 99.11             |
| Winter Wheat   | 0.95 | 67.16              | 58851 | 93.88             |
| Potatoes       | 0.85 | 65.8               | 6858  | 87.37             |
| Winter Barley  | 0.82 | 67.55              | 12092 | 96.6              |
| Alfalfa        | 0.81 | 59.86              | 2912  | 77.23             |
| Silage Corn    | 0.79 | 51.33              | 15921 | 74.17             |
| Flax           | 0.75 | 53.58              | 4123  | 60.42             |
| Peas           | 0.74 | 57.01              | 2112  | 86.84             |
| Fallow         | 0.66 | 42.43              | 11785 | 54.48             |

In this list there are three crop types where 0.5 > F1 < 0.75: Flax, Peas, and Fallow. Flax and Fallow are the weaker 2 of this group assessed by F1, Model Confidence, and Percent Agreement. Peas, like Flax and Fallow, had F1 <= 0.75 but its average confidence and percent agreement were much higher relatively. Looking at the confusion matrix now, it appears that the largest proportion of Pea's False positives and Flax's False positives were actually Flax (603) and Fallow (16) in ground truth respectively. So we have some considerable confusion among these three crop types, which would be good argument to re-train on all three of these crop types to reduce interclass confusion among the bottom 3 of our top-performers.

![cm](results/confusion_matrix.png)

**In summary, I would say that when looking on aggregate by crop type, the model underperforms in `u0c` AOI on this list of crop types below.**

| Crop           | F1   | Average Confidence | Count | Percent Agreement |
|----------------|------|--------------------|-------|-------------------|
| Winter Rye     | 0    |                    | 193   | 0                 |
| Spring Rye     | 0    |                    | 17    | 0                 |
| Millet         | 0    | 35.78              | 9     | 0                 |
| Beans          | 0    |                    | 336   | 0                 |
| Turnips        | 0    |                    | 11    | 0                 |
| Trees          | 0    |                    | 909   | 0                 |
| Spring Wheat   | 0.01 | 32.46              | 298   | 0.34              |
| Soybeans       | 0.07 | 54.01              | 33    | 66.67             |
| Vineyard       | 0.17 | 43.17              | 34    | 23.53             |
| Sorghum        | 0.18 | 42.4               | 105   | 14.29             |
| Triticale      | 0.19 | 47.79              | 311   | 45.98             |
| Clover         | 0.34 | 43.31              | 387   | 40.57             |
| Spring Oats    | 0.35 | 42.91              | 1131  | 30.59             |
| Winter Oats    | 0.39 | 47.11              | 184   | 40.22             |
| Spring Barley  | 0.44 | 46.85              | 4190  | 41.98             |
| Lentils        | 0.46 | 52.94              | 112   | 52.68             |
| Ryegrass       | 0.47 | 41.85              | 1885  | 53.05             |
| Grain Corn     | 0.61 | 51.13              | 6915  | 68.88             |
| Sunflowers     | 0.65 | 56.72              | 879   | 81.11             |
| Fallow         | 0.66 | 42.43              | 11785 | 54.48             |
| Peas           | 0.74 | 57.01              | 2112  | 86.84             |
| Flax           | 0.75 | 53.58              | 4123  | 60.42             |

| Total Count    | 35,959 |

This represents 23.3% (35,959) of the total ground truth dataset and deciding factors were a combination of F1, Agreement %, and Average Confidence.

## Prognosis :clipboard:

While it may be easier to filter this ground truth dataset by crop type using this list (and we could), it may be more helpful to be discerning at the individual field-level to compose a new training dataset out of the `u0c` AOI ground truth. By constructing a simple ruleset that defines whether or not to keep a field, we can filter the ground truth dataset to individual fields where the model underperforms rather than taking them all for a given crop type. It is most productive to re-train the model on fields where it was both:

1. Incorrect
2. Correct, but uncertain

This should allow a more pointed approach to active learning going forward.

Running `main.py` in `select` mode does this, applying these conditions and exporting a new .gpkg file (~83k records from provided gt dataset).

For the 'Correct, but uncertain' condition, we first compute the 10th,25th,50th,90th percentiles of model confidence by crop type. We then use the 50th percentile (median) of confidence in the filter for each crop type. This ensures that even for crop types where the model is making many correct predictions, we are still forcing the model in re-training to be even more confident on examples where it was less confident than usual, hopefully unlearning any bias toward specific spatio-temporal patterns that throw the model off.


### Architecture Considerations :mag:

If this is the L-TAE model from [Regrow's 2024 paper](https://www.mdpi.com/2073-445X/13/12/2246) I would be interested to see if there is a pattern between underperformance and field size. Since learned representations are derived from a pixel sample set (`S`) of fixed size, this means that depending on what `S` is, the learned representations of tiny fields passed into the encoder will be synthetically derived from a sample set `S` with duplicate pixels. Maybe it matters, maybe it doesn't, but it is an interesting architecture to read about and try to probe at!

From [Garnot 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.pdf): 
"A set S ⊂ [1, · · · , N] of S pixels is randomly drawn
from the N pixels within the parcel, as described in
Equation 1. When the total number of pixels in the
image is smaller than S, an arbitrary pixel is repeated"

![Garnot 2020 PSE-TAE](docs/Garnot%202020%20PSE-TAE.png)








