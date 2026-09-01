# Dataset

This project uses a three-class subset of the **CALTECH-101** dataset for single-object classification and bounding-box localization.

The selected classes are:

- Airplane
- Face
- Motorcycle

The dataset preparation and annotation format follow the PyImageSearch multi-class object detection and bounding-box regression tutorial.

## Dataset Sources

PyImageSearch tutorial:

https://pyimagesearch.com/2020/10/12/multi-class-object-detection-and-bounding-box-regression-with-keras-tensorflow-and-deep-learning/

Original CALTECH-101 dataset:

https://data.caltech.edu/records/mzrjq-6wc02

The full image dataset is not included in this repository. It can be obtained from the original source above.

## Expected Directory Structure

After preparing the dataset, organize it as:

```text
dataset/
├── images/
│   ├── airplane/
│   ├── face/
│   └── motorcycle/
│
└── annotations/
    ├── airplane.csv
    ├── face.csv
    └── motorcycle.csv
