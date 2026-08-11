# CIFAR-10 CNN from Scratch

A Convolutional Neural Network built from scratch using PyTorch for image classification on the CIFAR-10 dataset.

The project focuses on understanding the complete deep learning training pipeline rather than relying on high-level dataset or model implementations.

## Overview

The model takes 32×32 RGB images from CIFAR-10 and classifies them into one of 10 classes.

The project implements:

- Custom CIFAR-10 binary dataset parser
- Lazy image loading using `seek()` and `read()`
- Per-channel dataset normalization
- PyTorch `Dataset` implementation
- PyTorch `DataLoader`
- CNN architecture using `Conv2d`
- Batch Normalization
- ReLU activation
- Max Pooling
- Dropout
- Cross-Entropy Loss
- Adam optimizer
- Backpropagation
- Training accuracy calculation
- Test-set evaluation using `model.eval()`
- Model serialization using `state_dict`

## Dataset

The project uses the CIFAR-10 dataset.

CIFAR-10 contains:

- 50,000 training images
- 10,000 test images
- 10 classes
- RGB images
- Image size: `32 × 32`

The ten classes are:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
