import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset # for creating custom dataset
from torch.utils.data import DataLoader # for creating data loader
import os
from pathlib import Path

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths for .pt and .pth files
yolo_model_path = os.path.join(current_dir, "bestyolohandv8-noresize.pt")
xception_model_path = os.path.join(current_dir, "myXception.pth")

model_urls = {
    'xception':'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'
}

# Load YOLO hand detection model
yolo_model = YOLO(yolo_model_path)

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.3):
        super(Xception, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(dropout_rate)

        self.block1 = Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2 = Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3 = Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4 = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5 = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6 = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7 = Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8 = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9 = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10 = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11 = Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output
    
classification_model = Xception()
classification_model.load_state_dict(torch.load(xception_model_path, map_location='cpu'))
classification_model.eval()

# Define transformation for classification input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),               # Resize to match input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale input
])

# Mapping class IDs to letters
class_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

def detect_and_classify(image_path):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    print(f"Input image shape: {image.shape}")

    original_image = image.copy()

    # Convert image to RGB for YOLO
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform hand detection
    results = yolo_model(image_rgb, conf=0.1)  # Adjust confidence threshold if necessary

    # Access detections for the first image
    detections = results[0].boxes if results and results[0].boxes else []

    if not detections:
        print("No hands detected.")
        return

    for detection in detections:
        # Extract box coordinates, confidence, and class
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        confidence = detection.conf[0].item()
        class_id = detection.cls[0].item()

        print(f"Bounding box: ({x1}, {y1}, {x2}, {y2}), Confidence: {confidence}, Class ID: {class_id}")

        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Validate bounding box dimensions
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            print("Invalid bounding box, skipping...")
            continue

        # Crop the detected hand region
        cropped_image = image[y1:y2, x1:x2]

        # Convert to PIL Image for transformation
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        # Apply transformation
        input_tensor = transform(cropped_pil).unsqueeze(0)  # Add batch dimension

        # Perform classification
        with torch.no_grad():
            output = classification_model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            pred_letter = class_to_letter.get(pred_class, "?")

        print(f"Predicted Letter: {pred_letter}, Confidence: {confidence}")

        # Draw bounding box and prediction on the original image
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_image, f"{pred_letter} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert image to RGB for matplotlib
    result_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Display the final image with bounding box and predicted letter using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.axis('off')  # Hide axes for better visualization
    plt.title("Detected and Classified Sign Language")
    plt.show()

# Example usage
image_path = Path("infer/Test data/R.jpg")  # Relative path
detect_and_classify(str(image_path))