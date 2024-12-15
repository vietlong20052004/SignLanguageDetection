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
If you are interested in re-training the model, then run this command
```
pip install -r training_setup.txt
```
Then, proceed to run file infer.py if you are NOT interested in re-training the model. This loads the checkpoint (finalXception.pth), then feeds an image in the test set for it to predict.
However, if there is an error while running infer.py, then please use this kaggle link: https://www.kaggle.com/code/vinhsokaggle/dl-infer
