# Neural Image Compression for Gigapixel Histopathology Image Analysis

This repository contains links to code and data supporting the experiments described in the following paper:

```
D. Tellez, G. Litjens, J. van der Laak and F. Ciompi
Neural Image Compression for Gigapixel Histopathology Image Analysis
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019
DOI: 10.1109/TPAMI.2019.2936841
```
The paper can be accessed in the following link: https://doi.org/10.1109/TPAMI.2019.2936841

See ```featurize_patch_example.py``` for how to featurize a patch.

You can also use https://grand-challenge.org to featurize whole slides via ```run_nic_gc.py```.
For this you need an account capable of running algorithms and a token.
Contact the administrators for gaining access to these features.

Requirements: keras 2.2.4 and tensorflow 1.14
SimpleITK for converting the grandchallenge-created features to npy.

