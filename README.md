# Sign language classification
This repository attempts to apply the Xception model to classify sign language using the MNIST dataset. 
## 1. Sign language alphabet
Here is the sign language alphabet. Note: this is NOT the full version, as some characters are missing here.
![amer_sign2](https://github.com/user-attachments/assets/0c4a6838-18a6-4934-9430-7b0fb3120545)
## 2. Xception model
Here is the architecture of the Xception model. 
![c53a8928-1e70-4601-af2a-a4182fefd0d9](https://github.com/user-attachments/assets/d34c5986-5054-4626-af19-cb5e36c22dac)
## 3. How to run
First, you need to run this command:
```
pip install -r requirement.txt
```
Proceed to run file infer.py if you are NOT interested in re-training the model. This loads the checkpoint (finalXception.pth), then feeds an image in the test set for it to predict.
However, if there is an error while running infer.py, then please use this kaggle link: https://www.kaggle.com/code/vinhsokaggle/dl-infer

# YOLO
This repository attempts to apply the YOLO for hand detection training with the hand dataset. 
## 1. Hand dataset
The dataset is taken from the Hand Detection Dataset ( VOC / YOLO format ) on kaggle. Link: https://www.kaggle.com/datasets/nomihsa965/hand-detection-dataset-vocyolo-format
## 2. YOLOv8 model
The model used is YOLOv8 imported from the ultralytics library. Here is the architecture of the model: 
![14f43432-fea6-4660-b5dd-d68f68cd2162](https://github.com/user-attachments/assets/d0502c1c-de29-48f4-ac0e-e908bb2cb906)

# YOLO + Xception Workflow
This repository attempts to apply the YOLO for hand detection and then crop the image of the hand and feed it to the Xception model for classification. 
## 1. The workflow
Here is the graph for the full workflow:
![OurProcess](https://github.com/user-attachments/assets/6ef28251-9802-4246-a998-1364ab26f15a)

## 2. How to run
Proceed to run file infer.py. This loads the checkpoint (finalXception.pth) and the YOLO model (yolov8hand-noresize.pt), then feeds an image in the test set for it to predict. Change the image path to the path of the image to classify.
