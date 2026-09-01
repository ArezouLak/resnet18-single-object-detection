import torch


def box_iou_single(box1, box2):
    x1 = torch.maximum(box1[:,0], box2[:,0])
    y1 = torch.maximum(box1[:,1], box2[:,1])
    x2 = torch.minimum(box1[:,2], box2[:,2])
    y2 = torch.minimum(box1[:,3], box2[:,3])
    inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
    area1 = (box1[:,2]-box1[:,0]).clamp(min=0) * (box1[:,3]-box1[:,1]).clamp(min=0)
    area2 = (box2[:,2]-box2[:,0]).clamp(min=0) * (box2[:,3]-box2[:,1]).clamp(min=0)
    return inter / (area1 + area2 - inter).clamp(min=1e-8)
