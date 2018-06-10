# RevelationNet

## TODO
1. Generate adversarial images via Zeroth Order Optimization. 
2. The cleverhans version classifier only has about 78% accuracy, need to add learning rate decay to further increase performance.
3. Re-implement FGSM attack with cleverhans.

## Notes
- random multiplier on gradient hinders the convergence of FGSM attack.
- noise on convolutional kernel and especially negative noise greatly reduce classification accuracy.
- epsilon=0.02 and lr=0.003~0.01 is a suitable setting for FGSM attack on CIFAR-10 (96% rate of success).

## Contents
- `cleverhans`: A modified version of cleverhans.
- `hans_experiments/cifar10_classifier.py`: Tensorflow cifar10 classifier.
- `hans_experiments/experiments.py`: cleverhans experiments.
- `experiments/notebooks`: ipython notebook files.
- `experiments/FGSM.py`: Adversarial image generating module using modified Fast Gradient Sign Method (FGSM).
- `experiments/data_loader.py`: Dataset loading utilities, including `load_original_data` and `load_augmented_data`.
- `experiments/cifar10_classifier.py`: Keras cifar10 classifier.
- `experiments/classifiers.py`: All classifiers goes here.
- `experiments/experiments.py`: All experiments goes here. Experiments are wrapped into methods.
