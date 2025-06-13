# Potato Disease Classification

This project implements a deep learning model for classifying potato plant diseases using computer vision techniques. The model can identify different types of diseases in potato plants from images.

## Project Overview

The project uses a Convolutional Neural Network (CNN) to classify potato plant diseases. It's built using TensorFlow and Keras, and can identify multiple disease categories from plant images.

## Features

- Image-based disease classification
- Support for multiple disease categories
- Pre-trained model available for immediate use
- Data preprocessing and augmentation pipeline
- Model training and evaluation scripts

## Dataset

The dataset contains images of potato plants with different disease conditions. The images are organized into three classes:
- Early Blight
- Late Blight
- Healthy

## Model Architecture

The model uses a CNN architecture with the following key components:
- Input layer: 256x256x3 (RGB images)
- Data preprocessing layers:
  - Resizing
  - Rescaling
  - Random flip augmentation
  - Random rotation augmentation
- Convolutional layers for feature extraction
- Dense layers for classification

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Usage

1. Clone the repository:
```bash
git clone https://github.com/bhupesh-k-b/Potato_r.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Use the pre-trained model:
```python
import tensorflow as tf
model = tf.keras.models.load_model('model_potato.h5')
```

## Training

The model can be trained using the provided Jupyter notebook `potato_disease_classification.ipynb`. The notebook includes:
- Data loading and preprocessing
- Model architecture definition
- Training pipeline
- Evaluation metrics

## Model Files

- `model_potato.h5`: Pre-trained model weights
- `model.h5`: Alternative model weights

## Directory Structure

```
├── potato_disease_classification.ipynb  # Main training notebook
├── model_potato.h5                     # Pre-trained model
├── model.h5                           # Alternative model
├── DL/                                # Deep learning related files
└── Training/                          # Training data and scripts
```

## License

This project is open source and available under the MIT License.

## Author

Bhupesh K B

## Acknowledgments

- The dataset used for training
- TensorFlow and Keras communities
- Contributors and supporters of the project 