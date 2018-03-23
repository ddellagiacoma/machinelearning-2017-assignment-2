# Deep Neural Network

## 1. INTRODUCTION

The goal of this assignment is to experiment with neural networks, test the network using cross-validation over the training set through TensorFlow, one of the most used tools in machine learning. Then, train the network over the full training set and, finally, use the network to predict the examples in the test set.

This report summarizes the methodology used and the results obtained.

## 2. DATA

The data used to train and test the predictor refers to OCR (Optical Character Recognition) for handwritten characters from 16x8 bitmap images. The dataset is already split into training (41.721 instances) and test (10.431 instances) sets. Moreover, the labels of the training examples are known but the labels of the test set are hidden.

The train-data.csv and the test-data.csv contain 0-1 pixels of the 16x8 bitmap images (p_1_1, p_1_2, ..., p_1_8, p_2_1, ..., p_16_7, p_16_8) whereas the train-targets.csv file contains the labels of the training set, i.e., the 26 alphabet letters (a, b, ..., z).

The classifier has to classify the examples in the test set with higher accuracy than the reference baseline which is 0.75.

## 3. LEARNING

A deep neural network has been used to classify the handwritten characters. This deep neural network has been built using two convolutional layers alternated with 2 max pool layers, followed by a ReLU layer regularized with dropout, and finally a softmax layer to get the final predictions.

![image](https://user-images.githubusercontent.com/24565161/37827534-290bb4c0-2e98-11e8-972d-7ef769bb0212.png)

Convolutional layers are used to extract meaningful features from image. In particular, it divides a matrix into smaller patches and returns a number of features for each patch. The first convolutional layer extracts 32 features out of 5×5 patches whereas the second layer extracts 64 features for patch.

The max pool layers are filters used to reduce the number of the input dimensions between two convolutional layers. The first max pool halved the width and height of the images to 8x4 whereas the second layer to 4x2.

As learning algorithm, it has been used the Adam SGD algorithm with a learning rate of 10−4.

Finally, the network has been trained for 2600 epochs, each using a 50 examples mini-batch. The dropout keep probability is set to 0.5.

## 4. RESULTS

A 5-fold cross-validation has been used to test the predicting power of the network over the training set.

**Fold 1** | **Fold 2** | **Fold 3** | **Fold 4** | **Fold 5**
--- | --- | --- | --- | ---
**Mean accuracy** | 0.85524267 | 0.85402685 | 0.8565436 | 0.8575024 | 0.85438639


The cross-validation shows that the accuracy of the network is around 0.85.

Finally, the classifier has been trained over the full training set and has been used to predict the examples in the test-data.csv training set. However, the accuracy of the prediction cannot be tested because the labels of the test set are hidden. Nevertheless, the results of the cross-validation showed an higher accuracy than the reference baseline.

