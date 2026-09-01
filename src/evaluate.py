from pathlib import Path
import argparse, json
import cv2 as cv
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from network import MultiTaskResNet18
from utils import box_iou_single
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

def main(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes=json.loads(Path(args.classes).read_text()); df=pd.read_csv(args.test_csv)
    tfm=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    model=MultiTaskResNet18(resnet18(weights=ResNet18_Weights.DEFAULT),len(classes)); model.load_state_dict(torch.load(args.weights,map_location=device)); model=model.to(device).eval()
    yt,yp,ious=[],[],[]
    with torch.no_grad():
        for _,r in df.iterrows():
            img=cv.imread(r['image_path']);
            if img is None: continue
            img=cv.cvtColor(img,cv.COLOR_BGR2RGB); x=tfm(img).unsqueeze(0).to(device); logits,pb=model(x)
            tb=torch.tensor([[r['xmin'],r['ymin'],r['xmax'],r['ymax']]],dtype=torch.float32,device=device)
            yt.append(int(r['label_id'])); yp.append(logits.argmax(1).item()); ious.append(box_iou_single(pb,tb).item())
    report=classification_report(yt,yp,target_names=classes,digits=4); mean_iou=sum(ious)/len(ious) if ious else 0.0
    text=report+f'\nMean bounding-box IoU: {mean_iou:.4f}\n'; print(text); Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(text)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--test-csv',default='results/test.csv'); p.add_argument('--classes',default='results/classes.json'); p.add_argument('--weights',default='results/resnet18_multitask.pth'); p.add_argument('--output',default='results/evaluation/test_report.txt'); main(p.parse_args())
