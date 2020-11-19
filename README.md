# Neural Image Compression for Gigapixel Histopathology Image Analysis

This repository contains links to code and data supporting the experiments described in the following paper:

```
D. Tellez, G. Litjens, J. van der Laak and F. Ciompi
Neural Image Compression for Gigapixel Histopathology Image Analysis
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019
DOI: 10.1109/TPAMI.2019.2936841
```
The paper can be accessed in the following link: https://doi.org/10.1109/TPAMI.2019.2936841

This is a work in progress and more items will be added with time:

* Synthetic dataset. These images can be generated using code in the present repository (```synthetic_data_generation.py```) or directly downloaded from https://doi.org/10.5281/zenodo.3381498.
* Colorectal (CRC) dataset. To be announced soon.
* Compress a given whole-slide image. A whole-slide image can be compressed using code in the present repository (```featurize_wsi.py```) and pretrained models (```./models/encoders_patches_pathology/*.h5```).

works with keras 2.2.4 and tensorflow 1.14

