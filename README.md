# Overview of our deep learning internship assignment


## Description

each file is neural network assignment, we have 2 multiclassification models.
the coffee beans classification network is convolutional model and the Fashion MMIST is fully connected network.

## Getting Started

### Dependencies

To try working our projects in your IDE you need to install tensorflow and numpy to your IDE, these are already imported in the code.

# Fashion MMIST project:
We built neural networks using MNIST - Fashion data set from Keras data sets.
The data set consists of 70,000 images in a 28x28 grey scale.
The networks we built are all fully connected networks. one model for binary classification and for multi-classification (10 classes) we built one base model and 2 experiment models.
In the binary classification model:
Each image of X test and train contains 28x28 pixels, the pixels are integers between 0 and 255. 
We normalized X test and train by dividing by 255 to range between  0-1.

We selected two classes from the data set, "T-shirt/top" as 0, and "Trouser" as 1.
In addition, we also excluded 20 percent from the train data to perform as validation data in the training process.

### In the multi classification model:
We also excluded 20 percent from the train data to perform as a validation set in the training process.
Moreover, our network works on probability (0-1) and can't catch labels as numbers higher than 1.
Therefore, we used the method to-categorical(), to change the labels vector which has integers that represent different categories into a matrix that has binary values and columns equal to the number of categories in the data, in our case 10 categories.

Our approach was simple fully connected neural network models, for the binary classification we chose one neuron output which
represents probability to one class the network predicted and 1- this class probability that represents the second class by "sigmoid" activation.
For the multi-classification models, we chose an output layer with 10 neurons which represent probability to each one of the 10 classes probability, to get this complex probability we used "softmax" activation.

## Base Model
Base model architecture for multi-classification issue:
We chose a simple model fully connected structure with 2 hidden layers and "sigmoid" activation between all layers. the output is 1 neuron.

## Base model hyper parameters:
10 epochs, batches of 128 examples, and 20 percent validation set.
##First Experiment
First experiment architecture:
the difference between the base model architecture to this model is more hidden layers.
The input layer is the same, images pixel matrix 28x28 normalized to values between 1-0.

In this model we changed only the network architecture and remain the same hyper parameter to see which change will make the most difference - architecture or hyper parameter.

## Second Experiment
Second experiment architecture:
the difference between the base model to this is the hyper parameter.
Hence, this model architecture is the exact same as the base.

## Best model Results and Metrics
First, our base model and experiments models are on the same range of accuracy (87-89 percent).

The base model and second experiment have almost the same accuracy and loss for the train data and the validation set during the training process, while the second model has a 6 percent higher accuracy than the validation set during the training process.
Hence, we can assume that adding more epochs is better than adding more layers in order to prevent overfitting.

In addition, we ran the test set in all the models in order to evaluate each model's quality of predictions.
Base model evaluation:
86.9 percent accuracy.
Moreover, in the classification report we saw which classes our model has better predictions, due to class 6 "shirt" having the least better results with 0.67 precision and the best predictions in class 9 "ankle boot" having the best precision.

first experiment evaluation:
This experiment has the best accuracy of 89.8 percent.
The classification report is very similar to the base model report with the worst prediction for class 6 and the best prediction is for class 9.

second experiment evaluation:
88.1 percent accuracy.
The classification report is also very similar to the models above.

In the single prediction section, we ran one image pixel matrix as a batch in every model.
Each model returns a 1 x 10 vector in accordance with 10 classes. the model prediction for each class is valued as 1 in the matching index.
