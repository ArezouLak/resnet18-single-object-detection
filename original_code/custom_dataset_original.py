

from torch.utils.data import Dataset
import cv2 as cv
import torch


class Custom_dataset(Dataset):

    def __init__(self, tensors, transforms=None):

        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):

        images = self.tensors[0][index]
        labels = self.tensors[1][index]
        bboxs = self.tensors[2][index]


        if self.transforms:
            images = self.transforms(images)

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        bboxs = torch.tensor(
            bboxs,
            dtype=torch.float32
        )

        return images, labels, bboxs

    def __len__(self):

        return len(self.tensors[0])