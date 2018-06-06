# RevelationNet

## TODO
1. Generate adversarial images via FGSM and Zeroth Order Optimization. 
2. Convergence of model with innate noise.

## Contents
- `notebooks`: ipython notebook files
- `FGSM.py`: Adversarial image generating module using modified Fast Gradient Sign Method (FGSM).
- `data_loader.py`: Dataset loading utilies, including `load_original_data` and `load_augmented_data`.
- `cifar10_classifier.py`: Keras cifar10 classifier.
- `defense.py`: All models for defense goes here.
- `experiments.py`: All experiments goes here. Every experiment is wrapped into a method.
