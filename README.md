# Multi-channel PolyConvNet
## AI Hackathon - Challenge I
## Team Borides
Members: Kastan Day, Aria Coraor, Seonghwan Kim, Jiahui Yang, Ruijie Zhu

## Description
### This repo contains code to tackle challenge I of AI Hackathon. It is structured as follows:
.
├── data                                             # input data file for feature generation
├── Sliding_window_feature_generation.ipynb          # code used to generate multi-channel sliding window features
├── features                                         # folder containing all the features used to train the neural net
├── Multi-channel PolyConvNet.ipynb                  # code used to train / test the Multi-channel PolyConvNet
├── models                                           # folder containing the trained models
├── LICENSE
└── README.md


### Here we propose multi-channel polyconvnet, a 1D convolutional neural net for predicting the lamellar period of copolymers based on the sequence of beads. 


Features:
1. Sliding window features: 29-dimensional feature used to capture the activation of sequence
2. Kernels
- Exponential kernel: 30-dimensional feature used to capture the interaction decay at two ends
- Cosine kernel: used to capture the periodicity of the sequence
3. VAE features: 4-dimensional feature generated using the Variational Autoencoder model
4. Interaction parameter
