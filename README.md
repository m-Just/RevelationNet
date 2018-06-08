# RevelationNet

## TODO
1. Generate adversarial images via Zeroth Order Optimization. 

## Notes
- noise on convolutional kernel and negative noise greatly reduce classification accuracy.
- epsilon=0.02 and lr=0.003~0.01 is a suitable setting for FGSM attack on CIFAR-10 (96% rate of success).

## Contents
- `notebooks`: ipython notebook files.
- `FGSM.py`: Adversarial image generating module using modified Fast Gradient Sign Method (FGSM).
- `data_loader.py`: Dataset loading utilities, including `load_original_data` and `load_augmented_data`.
- `cifar10_classifier.py`: Keras cifar10 classifier.
- `classifiers.py`: All classifiers goes here.
- `experiments.py`: All experiments goes here. Experiments are wrapped into methods.
