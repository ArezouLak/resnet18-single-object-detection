import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
from torch.optim import Adam
from prepare_dataset import prepare_dataset
from custom_dataset import Custom_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from network import Network
from torchvision.models import resnet18, ResNet18_Weights


#get image paths, labels, bboxes
image_folder= "/home/arezou/practice/DL/project1_classification/vehicles/dataset/images"   
annot_folder="/home/arezou/practice/DL/project1_classification/vehicles/dataset/annotations"

images,labels, bboxes, imagepaths= prepare_dataset(annot_folder,image_folder)

#convert string labels to integer

le = LabelEncoder()
labels=le.fit_transform(labels)
print(le.classes_)


#devide 80% of dataset for training, 10% for vaidation and 10% for test

(train_images,rest_images,train_bboxes,rest_bboxes,train_labels,rest_labels,train_imgpaths,rest_imgpaths) = train_test_split(
    images,
    bboxes,
    labels,
    imagepaths,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

(val_images,test_images,val_bboxes,test_bboxes,val_labels,test_labels, val_imgpaths, test_imgpaths) = train_test_split(
    rest_images,
    rest_bboxes,
    rest_labels,
    rest_imgpaths,
    test_size=0.5,
    random_state=42,
    stratify=rest_labels
)

#save test dataset for later inference
test_df= pd.DataFrame({
    "test_imagepath" : test_imgpaths,
    "test_labelId": test_labels,
    "class_name" : le.inverse_transform(test_labels),
    "xmin": [box[0] for box in test_bboxes],

    "ymin": [box[1] for box in test_bboxes],

    "xmax": [box[2] for box in test_bboxes],

    "ymax": [box[3] for box in test_bboxes]
})



test_df.to_csv("/home/arezou/practice/DL/project1_classification/vehicles/test.csv", index=False)

#define device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



#define the mean and std of imagenet needed for applying normalization
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]
#define transforms
Transforms=transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])

#call function to get data in indices , read the images and apply transform to the images
trainds= Custom_dataset((train_images,train_labels,train_bboxes), Transforms)
valds=Custom_dataset((val_images,val_labels,val_bboxes), Transforms)


#load data in batch
BS=4
trainDs= DataLoader(trainds, BS, shuffle=True)
valDs=DataLoader(valds, BS, shuffle=False)


#define losses and optimizer and model
basemodel= resnet18(weights=ResNet18_Weights.DEFAULT)
model=Network(basemodel,num_classes=len(le.classes_))
model=model.to(device)

opt=Adam(model.parameters(), lr=0.0001)

classLoss=CrossEntropyLoss()
bboxLoss=SmoothL1Loss()  #MSELoss()


#intialize model history
H={"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
#start training for epochs
epochs=10

for e in range (epochs):

    #set model mode to train
    model.train()
    train_loss=0
    train_acc=0

    for image,label, bbox in trainDs:

        image= image.to(device)
        label=label.to(device).long()
        bbox=bbox.to(device).float()
        classpred,bboxpred=model(image)
        classloss=classLoss(classpred,label)
        bboxloss=bboxLoss(bboxpred, bbox)

        Tloss= (classloss+bboxloss)
        train_loss+= Tloss.item()
        train_acc+= (classpred.argmax(dim=1)==label).type(torch.float).sum().item()

        #zero previous graients , do backpropagation and update parameters
        opt.zero_grad()
        Tloss.backward()
        opt.step()

    
    #set the mode to evaluation
    model.eval()
    with torch.no_grad():
        val_loss=0
        val_acc=0

        for image,label, bbox in valDs:

            image= image.to(device)
            label=label.to(device).long()
            bbox=bbox.to(device).float()
            classpred,bboxpred=model(image)
            classloss=classLoss(classpred,label)
            bboxloss=bboxLoss(bboxpred, bbox)

            Vloss= (classloss+bboxloss)
            val_loss+= Vloss.item()
            val_acc+= (classpred.argmax(dim=1)==label).type(torch.float).sum().item()

    

    totalTLoss= train_loss / len(trainDs)
    totalTAC= train_acc/ len(trainds)
    totalVLoss= val_loss /len(valDs)
    totalVAC= val_acc /len(valds)

    H["train_loss"].append(totalTLoss)
    H["val_loss"].append(totalVLoss)
    H["train_acc"].append(totalTAC)
    H["val_acc"].append(totalVAC)

    print(f"for {e} epoch:")
    print( f"train loss : {totalTLoss} , train accuracy : {totalTAC}")
    print( f"val loss : {totalVLoss} , valaccuracy : {totalVAC}")



# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("/home/arezou/practice/DL/project1_classification/vehicles/dataset/plot.png")
# serialize the model to disk
torch.save(model.state_dict(), "/home/arezou/practice/DL/project1_classification/vehicles/dataset/weights.pth")  








