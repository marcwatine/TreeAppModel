# TreeApp Model Guide

## Content
This repository is a quick guide to using the species prediction model used in the TreeApp.
The main.py file is designed to have all the necessary ingredients to quickly understand the Input/Output of the pre-trained model

## Input
The expected model input is a (1, 224, 224, 3) numpy ndarray. The input pixels are scaled between [-1,1].
For that, we can load the image, for example with Pillow Image Library. We then have to convert said image to the array representation and expand its dimensionality.
Once this is done, we use keras.applications.mobilenet.preprocess_input to make sure we are indeed in the [-1,1] pixel represenation.

## Loading & Predicting
Load the model with keras.models.load_model and store it into a variable .
Then apply the predict method on the model and store it in a separate variable.

## Output
The output is (1, 60) numpy ndarray. It basically consists of the probability of the bark image pertaining to each of the 60 species that it has trained on.
To get the prediction, we take the highest probability species, by computing the argmax. 
Then we use the class_indices_file.json, which consists of a dictionary mapping species names to integers: "{Species names} --> [|0;59|]".
