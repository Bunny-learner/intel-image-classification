# Landscape Image Classification with Deep CNN

This project uses a deep Convolutional Neural Network (CNN) to classify landscape images into six different classes. The model is trained using data augmentation techniques and includes key layers like Batch Normalization (Batch Norm), ReLU, and Dropout to improve generalization and prevent overfitting.

## Model Architecture

- **Input Layer**: The input layer accepts landscape images.
- **Convolutional Layers**: Multiple convolutional layers for feature extraction.
- **Batch Normalization**: Used after each convolutional layer to normalize activations and speed up training.
- **ReLU Activation**: Non-linear activation function for introducing non-linearity into the model.
- **Dropout**: Applied after certain layers to prevent overfitting by randomly setting a fraction of input units to zero.
- **Fully Connected Layers**: The final classification layers that output the predicted class for each image.

## Data Augmentation

- **Augmentation Techniques**: Random flipping, rotation, and zooming to generate diverse variations of the dataset and prevent overfitting.

## Training

- **Epochs**: Trained over 30 epochs.
- **Loss Function**: Categorical Cross-Entropy Loss.
- **Optimizer**: Adam optimizer with a learning rate scheduler.
- **Metrics**: Achieved 80% accuracy on the test dataset.

## Results

- **Test Accuracy**: 80%
- **Classes**: The model classifies landscape images into six distinct classes.

## Installation

1. Clone this repository.
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model:
```bash
python train.py
