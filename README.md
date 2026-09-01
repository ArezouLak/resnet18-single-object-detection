# ResNet18 Single-Object Detection and Classification

This project adapts a pretrained **ResNet18** backbone into a multi-task network with two heads: one for **object classification** and one for **bounding-box regression**.

## Project Scope

The current dataset contains **one annotated object per image**, so the model predicts exactly one class and one bounding box for each image. This is a suitable formulation for single-object localization, but it is not a general multi-object detector.

For images containing multiple objects, multiple instances, or overlapping objects, a dedicated detector such as **YOLO**, Faster R-CNN, RetinaNet, or DETR would be more appropriate because these architectures can predict a variable number of detections per image.

## Architecture

```text
Input image
    |
    v
Pretrained ResNet18 backbone
    |
    v
Shared feature vector
    |
    +-------------------------+
    |                         |
    v                         v
Classification head      Bounding-box head
    |                         |
    v                         v
Class logits            xmin,ymin,xmax,ymax
```

The regression head ends with a sigmoid activation because bounding-box coordinates are normalized to `[0, 1]`.

## Classes

The original implementation contains three classes:

- airplane
- face
- motorcycle

## Training Objective

The model is trained jointly using:

- `CrossEntropyLoss` for classification
- `SmoothL1Loss` for bounding-box regression

```text
Total Loss = Classification Loss + Bounding Box Loss
```

The two contributions can also be weighted from the command line.

## Evaluation

The project reports both classification and localization performance:

- classification accuracy
- precision, recall, and F1-score
- mean bounding-box Intersection over Union (IoU)
- training and validation loss
- visual inference examples with true and predicted boxes

## Repository Structure

```text
.
├── src/
│   ├── network.py
│   ├── custom_dataset.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── data/
│   └── README.md
├── results/
│   ├── training_curves/
│   ├── inference/
│   └── evaluation/
├── models/
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

This project uses a small single-object detection dataset containing three classes:

- airplane
- face
- motorcycle

Each image contains one annotated object with one bounding box.

The dataset is included in this repository under:

`dataset/images/`

and the corresponding annotations are stored in:

`dataset/annotations/`
```

Each annotation CSV row should use:

```text
image_name,xmin,ymin,xmax,ymax,class_name
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train.py --annotation-dir dataset/annotations --image-dir dataset/images --epochs 10 --batch-size 4
```

## Evaluation

```bash
python src/evaluate.py
```

## Inference

```bash
python src/inference.py --num-images 10
```

Inference images show the ground-truth box and the predicted box together with the true and predicted class.

## Limitation and Future Work

This architecture predicts one object per image. A natural next step is to compare it with a **YOLO-based multi-object detector**, which can predict multiple object classes, confidence scores, and bounding boxes in the same image.

## Key Takeaway

This project demonstrates how a pretrained classification backbone can be extended into a simple multi-task localization network by attaching separate classification and bounding-box regression heads.
