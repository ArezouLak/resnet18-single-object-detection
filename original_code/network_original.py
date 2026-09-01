import torch
from torch.nn.modules import Module
from torch.nn import Linear, Dropout, BatchNorm2d, ReLU, Sigmoid
from torch.nn import Identity


class Network(Module):

    def __init__(self,basemodel, num_classes):

        super(Network,self).__init__()

        self.basemodel=basemodel
        self.num_classes=num_classes

        # Get number of ResNet features
        in_features = self.basemodel.fc.in_features
        self.basemodel.fc= Identity()
        
        #after the last fc layer of the basemodel(ResNet), make the netwmork double branch head, 
        #one for classification and one for detecting bbox(regression) 
        #define layers of classifier head
        self.classifier=torch.nn.Sequential(
             Linear(in_features, 512),
             ReLU(),
             Dropout(0.25),
             Linear( 512, 512),
             Dropout(0.5),
             Linear(512, self.num_classes)
        )

        # define layers of regressor  head to predict 4 coordinates of bboxes

        self.regressor= torch.nn.Sequential(
            Linear(in_features, 128),
            ReLU(),
            Linear(128 ,64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32,4),
            Sigmoid()  #our target bboxes are normalized to [0,1] Thats why Sigmoid is used here

        )

        

    def forward(self,x):

        features= self.basemodel(x)
        classlogits= self.classifier(features)
        bboxes= self.regressor(features)

        return classlogits, bboxes
    


# image
#   ↓
# ResNet backbone
#   ↓
# feature vector
#   ├──────────────→ classifier → class logits
#   │
#   └──────────────→ regressor  → bounding box