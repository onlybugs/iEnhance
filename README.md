# iEnhance: a Multi-scale Spatial Projection Encoding Network to Enhance Chromatin interaction data Resolution


## Summary

iEnhance is a multi-scale spatial projection and encoding network, to predict high-resolution chromatin interaction matrices from low-resolution and noisy input data. iEnhance can recover both short-range structural elements and long-range interaction patterns precisely. In addition, 

We provide the PyTorch implementations for both training and predicting procedures.

### **_Note:_** To explore the detailed architecture of the iEnhance please read the file _module.py_.


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


## Usage

## **_Note:_** Due to historical legacy issues, the **_from normga4 import Construct_** code statement in all training and prediction code is forbidden to be removed, otherwise it will cause the script to crash!.

### 1. Predicting
To execute the Hi-C matrix enhancement script, first configure the basic information of the script.

The following code blocks are the variables to be configured, **model** indicates the path where the pre-trained model is located, **fn** indicates the path of the input *.cool* format data, **chrs_list** is the chromosome number to be enhanced, and **cell_line_name** indicates the identifier of the final output result.
~~~python
from normga4 import Construct
from module import iEnhance

model = t.load("pretrained/BestHiCModule.pt",map_location = t.device('cpu'))
fn = "./HiCdata/Rao2014-K562-MboI-allreps-filtered.10kb.cool"
chrs_list = ['2' ,'4' ,'6' ,'8' ,'10' ,'12','16','17' ,'18','20','21']
cell_line_name = "K562"
~~~
