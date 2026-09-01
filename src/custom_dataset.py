import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, tensors, transforms=None):
        self.images, self.labels, self.bboxes = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        bbox = self.bboxes[index]
        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(bbox, dtype=torch.float32)

    def __len__(self):
        return len(self.images)
