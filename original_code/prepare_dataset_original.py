import cv2 as cv
import os
from glob import glob



def prepare_dataset(annot_folder,image_folder):

    imagepaths = []
    images=[]
    labels = []
    bboxes = []

    csv_files = glob(os.path.join(annot_folder, "*.csv"))

    for csv_file in csv_files:

        with open(csv_file, "r") as file:

            for line in file:

                imagepath, xmin, ymin, xmax, ymax, label = (
                    line.strip().split(",")
                )
                image_path=os.path.join(image_folder,label)
                image_final_path=os.path.join(image_path, imagepath)
              

                image = cv.imread(image_final_path)

                if image is None:
                    print("Could not read:", image_final_path)
                    continue

                height, width = image.shape[:2]

                xmin = float(xmin) / width
                xmax = float(xmax) / width

                ymin = float(ymin) / height
                ymax =float(ymax) / height

                image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image=cv.resize(image, (224,224))

                

                imagepaths.append(imagepath)
                images.append(image)
                labels.append(label)

                bboxes.append([
                    xmin,
                    ymin,
                    xmax,
                    ymax
                ])

    return images, labels, bboxes, imagepaths
        
    


            
image_folder= "/home/arezou/practice/DL/project1_classification/vehicles/dataset/images"   
annot_folder="/home/arezou/practice/DL/project1_classification/vehicles/dataset/annotations"
images, labels, bboxes, imagepaths= prepare_dataset(annot_folder,image_folder)
print(len(images))
print(len(imagepaths))
print(len(labels))
print(len(bboxes))