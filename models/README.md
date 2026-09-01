# Model Weights

This project trains a multi-task ResNet18 model with two prediction heads:

- a classification head for object class prediction
- a regression head for bounding-box localization

The training script saves the trained checkpoint as:

`resnet18_multitask.pth`

Trained `.pth` checkpoint files are excluded from the main repository to keep the repository lightweight.
