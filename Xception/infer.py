import pandas as pd
import matplotlib.pyplot as plt
import random
import string

import torch
from torch.utils.data import Dataset
from model_building import Xception

character_mapping = {}
alphabet = list(string.ascii_uppercase)
for count, ch in enumerate(alphabet):
    character_mapping[count] = ch

class Generator(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def main():
    test_data = pd.read_csv("Dataset/sign_mnist_test.csv")
    (x_test, y_test) = test_data.drop(labels = 'label', axis = 1), test_data['label']

    test_data = Generator(x_test, y_test)

    model = Xception()
    model.load_state_dict(torch.load('myXception.pth'))
    model.eval()

    random_index = random.randint(0, len(x_test) - 1)

    random_image = x_test[random_index]
    true_label = y_test[random_index]

    plt.imshow(random_image.squeeze(), cmap="gray")
    plt.title(f"True Label: {true_label}")
    plt.axis("off")
    plt.show()

    random_image_tensor = torch.tensor(random_image).unsqueeze(0).permute(0, 3, 1, 2) 

    # Make prediction
    with torch.no_grad():
        output = model(random_image_tensor)
        prediction = torch.argmax(output, dim=1).item()

    print(f"Predicted Label: {prediction}")
    print(f"Predicted Character: {character_mapping[prediction]}")

if __name__ == "__main__":
    main()