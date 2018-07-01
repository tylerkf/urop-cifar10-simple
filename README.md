# Simple CIFAR-10 Neural Network for UROP 2018
Simple convolutional neural network for my UROP.

### Training
Run train.py with the model path as a parameter.
Default path is '/tmp/cifar10_model'.

### Prediction
Run predict.py with the image path as its first parameter (must be 32x32 RGB) and model path as its second.

## Model
Simple CNN structure with entire 32x32 image as input and no pre processing on the dataset.
Model structure and parameters are taken from [this tensorflow tutorial](https://www.tensorflow.org/tutorials/deep_cnn).
