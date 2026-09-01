from pathlib import Path
import argparse, json
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from custom_dataset import CustomDataset
from network import MultiTaskResNet18
from prepare_dataset import prepare_dataset
from utils import box_iou_single

MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

def main(args):
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    images,labels,bboxes,paths=prepare_dataset(args.annotation_dir,args.image_dir)
    le=LabelEncoder(); labels=le.fit_transform(labels)
    train_i,rest_i,train_b,rest_b,train_l,rest_l,train_p,rest_p=train_test_split(images,bboxes,labels,paths,test_size=0.2,random_state=42,stratify=labels)
    val_i,test_i,val_b,test_b,val_l,test_l,val_p,test_p=train_test_split(rest_i,rest_b,rest_l,rest_p,test_size=0.5,random_state=42,stratify=rest_l)
    pd.DataFrame({'image_path':test_p,'label_id':test_l,'class_name':le.inverse_transform(test_l),'xmin':[b[0] for b in test_b],'ymin':[b[1] for b in test_b],'xmax':[b[2] for b in test_b],'ymax':[b[3] for b in test_b]}).to_csv(out/'test.csv',index=False)
    (out/'classes.json').write_text(json.dumps(le.classes_.tolist(),indent=2))
    tfm=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    train_ds=CustomDataset((train_i,train_l,train_b),tfm); val_ds=CustomDataset((val_i,val_l,val_b),tfm)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True); val_loader=DataLoader(val_ds,batch_size=args.batch_size)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MultiTaskResNet18(resnet18(weights=ResNet18_Weights.DEFAULT),len(le.classes_)).to(device)
    opt=Adam(model.parameters(),lr=args.lr); cls_loss_fn=CrossEntropyLoss(); box_loss_fn=SmoothL1Loss()
    H={k:[] for k in ['train_loss','train_acc','train_iou','val_loss','val_acc','val_iou']}
    for epoch in range(args.epochs):
        model.train(); loss_sum=correct=iou_sum=0
        for x,y,b in train_loader:
            x,y,b=x.to(device),y.to(device),b.to(device); logits,pb=model(x)
            loss=args.classification_weight*cls_loss_fn(logits,y)+args.bbox_weight*box_loss_fn(pb,b)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum+=loss.item(); correct+=(logits.argmax(1)==y).sum().item(); iou_sum+=box_iou_single(pb.detach(),b).sum().item()
        model.eval(); vloss=vcorrect=viou=0
        with torch.no_grad():
            for x,y,b in val_loader:
                x,y,b=x.to(device),y.to(device),b.to(device); logits,pb=model(x)
                loss=args.classification_weight*cls_loss_fn(logits,y)+args.bbox_weight*box_loss_fn(pb,b)
                vloss+=loss.item(); vcorrect+=(logits.argmax(1)==y).sum().item(); viou+=box_iou_single(pb,b).sum().item()
        vals=[loss_sum/len(train_loader),correct/len(train_ds),iou_sum/len(train_ds),vloss/len(val_loader),vcorrect/len(val_ds),viou/len(val_ds)]
        for k,v in zip(H,vals): H[k].append(v)
        print(f'Epoch {epoch+1}/{args.epochs} | train loss={vals[0]:.4f} acc={vals[1]:.4f} IoU={vals[2]:.4f} | val loss={vals[3]:.4f} acc={vals[4]:.4f} IoU={vals[5]:.4f}')
    torch.save(model.state_dict(),out/'resnet18_multitask.pth'); pd.DataFrame(H).to_csv(out/'training_history.csv',index=False)
    plt.figure(); plt.plot(H['train_loss'],label='train_loss'); plt.plot(H['val_loss'],label='val_loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout(); plt.savefig(out/'loss_curve.png'); plt.close()
    plt.figure(); plt.plot(H['train_acc'],label='train_accuracy'); plt.plot(H['val_acc'],label='val_accuracy'); plt.plot(H['train_iou'],label='train_iou'); plt.plot(H['val_iou'],label='val_iou'); plt.xlabel('Epoch'); plt.ylabel('Metric'); plt.legend(); plt.tight_layout(); plt.savefig(out/'metrics_curve.png'); plt.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--annotation-dir',default='dataset/annotations'); p.add_argument('--image-dir',default='dataset/images'); p.add_argument('--output-dir',default='results'); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--classification-weight',type=float,default=1.0); p.add_argument('--bbox-weight',type=float,default=1.0); main(p.parse_args())
