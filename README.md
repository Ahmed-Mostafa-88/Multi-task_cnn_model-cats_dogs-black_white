# Custom Image Classification with CNN using OpenCV on Python

## Visual Representation for
### Data augmentation
[Your Visual Representation for Data Augmentation]

### Model Architecture
[Your Visual Representation for Model Architecture]

### Model Prediction
[Your Visual Representation for Model Prediction]

## Table of Contents
1. [Introduction](#introduction)
    1. [Project Title](#project-title)
    2. [Project Goals](#project-goals)
2. [Data Creation](#data-creation)
3. [Data Preprocessing](#data-preprocessing)
    1. [Data Loading and Structuring](#data-loading-and-structuring)
    2. [Data Resizing, Normalization, Augmentation](#data-resizing-normalization-augmentation)
    3. [Train-Validation-Test Split](#train-validation-test-split)
    4. [One-Hot Encoding](#one-hot-encoding)
4. [CNN Model](#cnn-model)
    1. [Input Layer](#input-layer)
    2. [Convolutional Layers](#convolutional-layers)
    3. [Batch Normalization](#batch-normalization)
    4. [Max Pooling Layers](#max-pooling-layers)
    5. [Flatten Layer](#flatten-layer)
    6. [Dense Layers](#dense-layers)
5. [Model Training](#model-training)
6. [Results](#results)

## Introduction
### Project Title
Custom animal type (cat/dog) and color (white/black) Classification with CNN using OpenCV on Python.

### Project Goals
Develop an accurate CNN model capable of classifying images into cat or dog categories and identifying their colors white or black with high precision and recall.

## Data Creation
The dataset consists of images categorized into cats and dogs, each divided into white and black color categories. Details:
- Cats - White: 400 images
- Cats - Black: 400 images
- Dogs - White: 400 images
- Dogs - Black: 400 images

## Data Preprocessing
### Data Loading and Structuring
- Images from categories "cats" and "dogs" are loaded.
- Each image is associated with labels indicating whether it's a cat or a dog, as well as its color (black or white).

### Data Resizing, Normalization, Augmentation
Techniques | Explanation | Display
--- | --- | ---
Resizing | Resize images to (150x150 pixels) for the model's input size. | [Your Visual Representation for Resizing]
Normalization | Normalize pixel values to a scale between 0 and 1 for effective pattern learning. | [Your Visual Representation for Normalization]
Rotation, Width/Height Shift, Horizontal Flip | Augment data for better training and reduce overfitting. | [Your Visual Representation for Augmentation]

- Augmented images are combined with original images for a more diverse training dataset.

### Train-Validation-Test Split
- Training set: Used to train the CNN model.
- Validation set: Fine-tuning model parameters and avoiding overfitting.
- Test set: Evaluating model performance on unseen data.

### One-Hot Encoding
Categorical labels for dog/cat and color classifications transformed into binary vectors for effective training.

## CNN Model
The CNN model architecture involves:
### Input Layer
- Input Shape (150, 150, 3): Images with dimensions of 150x150 pixels and three channels (RGB).

### Convolutional Layers
- Three convolutional layers: 32 filters (3x3), 64 filters (3x3), 128 filters (3x3) with ReLU activation.

### Batch Normalization
- Applied after each convolutional layer to normalize activations for stability.

### Max Pooling Layers
- Downsampling spatial dimensions by a factor of (2, 2) after each pair of convolutional layers.

### Flatten Layer
- Flattens output from convolutional layers into a one-dimensional array for dense layers.

### Dense Layers
- Dense layer with 512 units, ReLU activation, and Dropout (rate: 0.5) to prevent overfitting.
- Dog/Cat Classification:
  - Dense Layer (256 units) with ReLU activation, Dropout (rate: 0.5), and Output layer (2 units, softmax activation).
- Color Classification:
  - Dense Layer (128 units) with ReLU activation, Dropout (rate: 0.5), and Output layer (2 units, softmax activation).

## Model Training
- Optimizer Initialization: Learning rate control and exponential decay schedule for optimization.
- Model Compilation: RMSprop optimizer, specified loss functions, and evaluation metric (accuracy).
- Callback Setup: Early Stopping and Model Checkpointing for better training.
- Model Training: Trained on the training data for a specified number of epochs and batch size.

## Results
- Training Results:
  - Accuracy: The accuracy for both dog/cat classification and color classification tasks improves gradually over epochs. Initially, the accuracy for dog/cat tasks starts around 54%-55% and progressively increases, while color starts high and continue increasing.
  - Loss: The overall loss (including both dog/cat and color losses) decreases significantly from an initial value of around 6.97 to a much lower value, indicating improvement in the model's performance.
- Validation Results:
  - Validation loss and accuracy follow similar patterns to the training set, indicating the model's consistency in learning and generalizing.
  - Dog/Cat Accuracy on Validation: varies around 52% to 80% 
  - Color Accuracy on Validation: Stays consistently high.
- Testing Results:
  - High accuracy in both dog/cat (79.22%) and color (98.85%) classifications, showcasing strong performance on test data.
- Classification Reports:
  - Precision, Recall, F1-Score, and Accuracy metrics for both dog/cat and color classifications.
