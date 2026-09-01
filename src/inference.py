from pathlib import Path
import argparse, json, random
import cv2 as cv
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from network import MultiTaskResNet18
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

def main(args):
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes=json.loads(Path(args.classes).read_text()); df=pd.read_csv(args.test_csv)
    model=MultiTaskResNet18(resnet18(weights=ResNet18_Weights.DEFAULT),len(classes)); model.load_state_dict(torch.load(args.weights,map_location=device)); model=model.to(device).eval()
    tfm=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    for i in random.sample(range(len(df)),min(args.num_images,len(df))):
        r=df.iloc[i]; img=cv.imread(r['image_path']);
        if img is None: continue
        h,w=img.shape[:2]; rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB); x=tfm(rgb).unsqueeze(0).to(device)
        with torch.no_grad(): logits,pb=model(x)
        pred=classes[logits.argmax(1).item()]; conf=torch.softmax(logits,1).max().item(); p=pb[0].cpu().numpy()
        pred_box=(int(p[0]*w),int(p[1]*h),int(p[2]*w),int(p[3]*h)); true_box=(int(r['xmin']*w),int(r['ymin']*h),int(r['xmax']*w),int(r['ymax']*h))
        display=img.copy(); cv.rectangle(display,true_box[:2],true_box[2:],(0,255,0),2); cv.rectangle(display,pred_box[:2],pred_box[2:],(0,255,255),2)
        cv.putText(display,f"True: {r['class_name']}",(20,30),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2); cv.putText(display,f'Pred: {pred} ({conf:.1%})',(20,60),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv.imwrite(str(out/f'prediction_{i}.jpg'),display)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--test-csv',default='results/test.csv'); p.add_argument('--classes',default='results/classes.json'); p.add_argument('--weights',default='results/resnet18_multitask.pth'); p.add_argument('--output-dir',default='results/inference'); p.add_argument('--num-images',type=int,default=10); main(p.parse_args())
