# iEnhance: a Multi-scale Spatial Projection Encoding Network to Enhance Chromatin interaction data Resolution


## Summary

iEnhance is a multi-scale spatial projection and encoding network, to predict high-resolution chromatin interaction matrices from low-resolution and noisy input data. iEnhance can recover both short-range structural elements and long-range interaction patterns precisely. In addition, 

We provide the PyTorch implementations for both training and predicting procedures.


## Dependency

iEnhance is written in Python3 with PyTorch framework.

The following versions are recommended when using iEnhance:

- torch 1.12.1
- numpy 1.21.2
- scipy 1.7.1
- pandas 1.3.3
- scikit-learn 1.0.2
- matplotlib 3.1.0
- tqdm 4.62.3
- cooler 0.8.11

**_Note:_** GPU usage for training and testing is highly recommended.


## Data Preparation

### 1. Hi-C data

For Hi-C data, we desire an input in _.cool_ file format. If your data is in _.hic_ format, please use the format conversion tool to convert to _.cool_ file format.

### 2. Micro-C and scHi-C data

For single-cell Hi-C data, we also desire to enter a separate _.cool_ file format. If your data is in another format for storing scHi-C data, please convert to _.scool_ format. Then use the cooler toolkit to pool _.scool_ file into multiple individual _.cool_ files for subsequent enhancements. For Micro-C data, please use the same pre-processing method as for Hi-C data.
