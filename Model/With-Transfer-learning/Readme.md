# Image Caption Generator
Make project using Transfer learning we use VGG16 model for pre tained model use for next layer we use Filcker8k image datasets

```
import tensorflow 
import os
import pickle
import numpy as np
# %pip install tqdm
from tqdm import tqdm

```
Use these library to import and tqdm is show loading simble so use that library
tqdm:-progress bar dikhane ke liye (sirf cosmetic, processing speed pe koi effect nahi).

### Important library 
``` 
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
````
load_img that is method to load image load_img → PIL Image object deta hai.

## Download latest version
i am use the Kaggle Datasets use use line of code not dowload the datasets useing kaggle account and APIs
```
import kagglehub
path = kagglehub.dataset_download("adityajn105/flickr8k")

print("Path to dataset files:", path)
```
### Make Model use VGG16

```
#load Model
model = VGG16()
#Restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
#summarize the model
model.summary()
```
This is a Convolutional Neural Network which is pre-trained on ImageNet dataset (1.2M images, 1000 classes).

By default, VGG16() gives you the complete model (input → convolutional blocks → fully connected layers → final 1000 class softmax output).
👉 Here you cut the original VGG16 model.
```
model.layers[-2] means second last layer.
```
The last layer of VGG16 is: Dense(1000, activation='softmax') → which predicts 1000 classes of ImageNet.

It has a first layer: Dense(4096, activation='relu') → a feature representation layer.

So you said:

"I don't want the final classification, I want the features that are generated just before the last softmax."

Result → now the model will take the input image and output a 4096-dimensional feature vector, not 1000 classes of ImageNet.

This will print the Structure of the model.You See:

Input:(244,244,3) (image size for VGG16).

Output:(4096,)(features vector)