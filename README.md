# Custom Image Classification with CNN using OpenCV on Python

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
7. [Model Prediction](#model-prediction)

## Introduction
### Project Title
Custom animal type (cat/dog) and color (white/black) Classification with CNN using OpenCV on Python.

### Project Goals
Develop an accurate CNN model capable of classifying images into cat or dog categories and identifying their colors white or black with high precision and recall.

## Data Creation

## About Dataset
### Description
This dataset comprises a collection of images that are categorized into two main classes: cats and dogs. Each class further contains subcategories based on color, namely white and black.

The dataset includes the following structure:

- **Cats**
  - **White**: Images of cats with white fur.
  - **Black**: Images of cats with black fur.
  
- **Dogs**
  - **White**: Images of dogs with white fur.
  - **Black**: Images of dogs with black fur.

The dataset is well-structured and labeled, making it suitable for tasks like image classification. It is particularly useful for training and evaluating models designed to classify animal types (cats or dogs) and distinguish their colors (white or black).

### Contents

- **Cats (White)**: [Number of images: 400]
- **Cats (Black)**: [Number of images: 400]
- **Dogs (White)**: [Number of images: 400]
- **Dogs (Black)**: [Number of images: 400]

These images are sourced from diverse collections, ensuring a wide variety of cat and dog breeds across various backgrounds and environments. The dataset's labeling and diversity provide a robust foundation for developing and testing image classification models.

### Dataset Creation

The dataset was created by me and is available on Kaggle at [CD-Deep Dataset](https://www.kaggle.com/datasets/ahmedmostafa11111/cd-deep). 


## Data Preprocessing
### Data Loading and Structuring
- Images from categories "cats" and "dogs" are loaded.
- Each image is associated with labels indicating whether it's a cat or a dog, as well as its color (black or white).

### Data Resizing, Normalization, Augmentation
Techniques | Explanation | Display
--- | --- | ---
Resizing | Resize images to (150x150 pixels) for the model's input size. | ![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/3d4b3486-6fbf-4bcc-88e0-ea30be634e5d)
Normalization | Normalize pixel values to a scale between 0 and 1 for effective pattern learning. | ![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/2771e846-30f7-4281-8204-e9f0ac765925)
Rotation | Rotate the image by a certain angle (20). | ![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/fa178371-d476-4b17-95aa-f687a54891a0)
Width/Height Shift, Horizontal Flip | Perform shifting of the image horizontally and vertically. | ![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/821c3259-c734-4992-b295-9b6cb63b6b97)![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/e40441cb-369d-4d06-9bd6-b6cfd719d691)
Horizontal Flip | Flip the image horizontally. | ![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/c203856d-5225-4a3b-88ac-fd5b18763f68)

- Augmented images are combined with original images for a more diverse training dataset.

### Train-Validation-Test Split
- Training set: Used to train the CNN model.
- Validation set: Fine-tuning model parameters and avoiding overfitting.
- Test set: Evaluating model performance on unseen data.

### One-Hot Encoding
Categorical labels for dog/cat and color classifications transformed into binary vectors for effective training.

## CNN Model

### Model Architecture
![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/a7601275-5ad6-4488-837a-e0cbf12fbb4f)

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
![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/dd77acfc-91f9-44f5-9013-457cfcfb3fa4)
- Training Results:
  - Accuracy: The accuracy for both dog/cat classification and color classification tasks improves gradually over epochs. Initially, the accuracy for dog/cat tasks starts around 54%-55% and progressively increases, while color starts high and continue increasing.
  - Loss: The overall loss (including both dog/cat and color losses) decreases significantly from an initial value of around 6.97 to a much lower value, indicating improvement in the model's performance.
- Validation Results:
  - Validation loss and accuracy follow similar patterns to the training set, indicating the model's consistency in learning and generalizing.
  - Dog/Cat Accuracy on Validation: varies around 52% to 80% 
  - Color Accuracy on Validation: Stays consistently high.
- Testing Results:
  - High accuracy in both dog/cat (79.22%) and color (98.85%) classifications, showcasing strong performance on test data.
## Results
After training the model, here are the evaluation metrics for both dog/cat and color classifications:

### Dog/Cat Classification
![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/89a15866-97d4-4a73-bb59-5aded6eacfb0)
- **Precision and Recall**:
  - Class 0 (Dog): Precision - 0.90, Recall - 0.67
  - Class 1 (Cat): Precision - 0.72, Recall - 0.92
  - Interpretation:
    - The model correctly identifies dogs around 90% of the time, while for cats, it is correct about 72% of the time.
    - The model correctly identified 67% of actual dogs and 92% of actual cats in the dataset.
- **F1-Score**:
  - Class 0 (Dog): 0.77
  - Class 1 (Cat): 0.81
  - Interpretation:
    - F1-scores (0 to 1) suggest a somewhat balanced performance between precision and recall for both dog and cat categories.
- **Accuracy**:
  - Overall accuracy: 79%
  - Interpretation:
    - The model correctly predicted approximately 79% of the images as either dogs or cats in the test set.

### Color Classification
![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/f4cc7ab0-5641-4f2c-9dbb-098241aae5fd)
- **Precision and Recall**:
  - Class 0 (Black): Precision - 0.98, Recall - 0.99
  - Class 1 (White): Precision - 0.99, Recall - 0.99
  - Interpretation:
    - The model accurately predicts black or white color around 99% of the time.
    - It correctly identifies about 99% of the actual black and white images in the dataset.
- **F1-Score**:
  - Class 0 (Black): 0.99
  - Class 1 (White): 0.99
  - Interpretation:
    - High F1-scores (0 to 1) indicate a high level of accuracy in identifying both black and white colors, suggesting a balance between precision and recall.
- **Accuracy**:
  - Overall accuracy: 99%
  - Interpretation:
    - The model correctly classified 99% of the images with respect to their colors, whether black or white.

### Model Prediction

| Category      | Image Prediction |
|---------------|------------------|
| White Dog     | ![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/e73c1db5-d3af-44c2-b41d-e18a7b969690) |
| Black Dog     | ![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/ed309894-a738-48a4-876c-b030234a858c) |
| White Cat     | ![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/866daebc-70e4-4154-829f-4339c494068d) |
| Black Cat     | ![image](https://github.com/Ahmed-Mostafa-88/Multi-task_cnn_model-cats_dogs-black_white/assets/144740078/72fdb210-e3dd-4ef2-958e-10ac7c30dc5e) |

