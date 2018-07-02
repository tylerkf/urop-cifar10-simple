# Simple CIFAR-10 Neural Network for UROP 2018
Simple convolutional neural network for my UROP. Built using the tensorflow's estimator API.

### Setup
Run `pip install -r requirements.txt`.

All code was developed and tested with Python 3.6 on macOS High Sierra.

### Training
Run `train.py` with the model path as a parameter.

Default path is `/tmp/cifar10_model`.

### Prediction
Run `predict.py` with the mode directory as its first parameter (optional) and image path as its second (must be 32x32 RGB).

`./model` contains a pre-trained model with 70% accuracy

## Model
Model structure and parameters are taken from [this tensorflow tutorial](https://www.tensorflow.org/tutorials/deep_cnn).