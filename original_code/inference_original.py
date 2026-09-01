import torch
import os
import cv2 as cv
from torchvision import transforms
import random
from torchvision.models import resnet18, ResNet18_Weights
from network import Network



#read cvs file for test dataset and get the true class names and bboxes
csv_file="/home/arezou/practice/DL/project1_classification/vehicles/test.csv"
image_folder= "/home/arezou/practice/DL/project1_classification/vehicles/dataset/images" 
output_folder="/home/arezou/practice/DL/project1_classification/vehicles/inference_folder"

with open (csv_file,"r") as file:
    images=[]
    labelIDs=[]
    class_names=[]
    bboxes=[]


    heading= file.readline()

    for line in file:

        imgpath, labelID, class_name, xmin, ymin, xmax,ymax= line.strip().split(",")  #strings

        image_path=os.path.join(image_folder,class_name)
        image_final_path= os.path.join(image_path,imgpath)

        image=cv.imread(image_final_path)
        image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image=cv.resize(image, (224,224))
        images.append(image)
        labelIDs.append(int(labelID))
        class_names.append(class_name)
        bboxes.append([float(xmin),float(ymin),float(xmax),float(ymax)])


#print(len(images))
        
#define the mean and std of imagenet needed for applying normalization
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]
#define the same transforms applied to the train images
Transforms=transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])


#define device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the saved weights of the model
basemodel=resnet18(weights=ResNet18_Weights.DEFAULT)
model=Network(basemodel, num_classes=3)


weight_path="/home/arezou/practice/DL/project1_classification/vehicles/weights.pth"

model.load_state_dict(torch.load(weight_path,map_location=device))
model=model.to(device)
model.eval()
# get the 10 random images from test set
random_indices=random.sample(range(len(images)),10)

Classes=["airplane","face","motorcycle"]
#loop through the selected test images

with torch.no_grad():

    for i in random_indices:
        
        image_rgb=images[i]
        true_bbox=bboxes[i]
        true_class_name=class_names[i]
        xmint,ymint,xmaxt,ymaxt=true_bbox
        xmint = int(xmint * 224)
        ymint = int(ymint * 224)
        xmaxt = int(xmaxt * 224)
        ymaxt = int(ymaxt * 224)
        #apply transform
        image_tensor=Transforms(image_rgb)
        #add batch dimension(1)
        image_tensor=image_tensor.unsqueeze(0)
        image_tensor=image_tensor.to(device)

        #pass the image to the model
        classlogits,bboxpred=model(image_tensor)
        xminp, yminp, xmaxp, ymaxp= bboxpred[0].cpu().numpy()  #[batch_size, 4] [1, 4]
        xminp = int(xminp * 224)
        yminp = int(yminp * 224)
        xmaxp = int(xmaxp * 224)
        ymaxp = int(ymaxp * 224)
        pred_index=classlogits.argmax(dim=1).item()
        pred=Classes[pred_index]

        text1=f"true:{true_class_name}"
        text2=f"prd:{pred}"
        display_image = cv.cvtColor(image_rgb.copy(),cv.COLOR_RGB2BGR)

        cv.putText(
            display_image, text1, (20,35), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2
        )
    
        cv.putText(
            display_image, text2, (20,70), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255), 2
        )

        cv.rectangle(display_image, (xmint,ymint),(xmaxt,ymaxt),(0,225,0),2)
        cv.rectangle(display_image, (xminp,yminp),(xmaxp,ymaxp),(0,225,255),2)

        output_path=os.path.join(output_folder, f"prediction_{i}.jpg")
        cv.imwrite(output_path, display_image)
    






