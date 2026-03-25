# ecg-arrhythmia-model-comparison


## Overview
This project focuses on classification of ECG heartbeats using multiple deep learning architectures and performing a comparative analysis of their performance.

The goal is to evaluate how different models — ranging from basic neural networks to advanced architectures with attention — perform on ECG signal classification.

---

## Objectives
- Perform ECG beat classification using multiple deep learning models
- Handle class imbalance in medical data
- Compare performance across architectures
- Analyze strengths of CNN, RNN, and hybrid models
- Explore feature extraction using pretrained networks

---

## Dataset
- MIT-BIH Arrhythmia Dataset
- Source: PhysioNet 
- Contains annotated ECG signals for different arrhythmia classes

### Preprocessing Steps
- Signal segmentation into heartbeat windows
- Normalization (Min-Max scaling)
- Label encoding
- Train-test split (stratified)
- Class imbalance handled using class weights

---

## Models Implemented

### 1. Multi-Layer Perceptron (MLP)
- Fully connected neural network
- Used as a baseline model

### 2. Convolutional Neural Network (CNN)
- Extracts spatial features from ECG signals
- Includes Conv1D, Batch Normalization, and Dropout

### 3. Pretrained CNN Feature Extraction
- ResNet
- MobileNet
- Used for extracting high-level features

### 4. RNN-based Models
- RNN
- LSTM
- GRU
- Used for sequential modeling of ECG signals


### 5. Attention-based LSTM
- Self-attention applied on LSTM outputs
- Helps focus on important timesteps

---

## Training Configuration
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy, Confusion Matrix
- Regularization: Dropout, Early Stopping

---

## Evaluation Metrics
- Accuracy
- Confusion Matrix
- Loss vs Accuracy Curves
- Class-wise performance

---

## Key Insights
- CNN performs best for spatial feature extraction
- RNN/LSTM/GRU capture temporal dependencies
- Hybrid models improve representation
- Attention improves performance and interpretability
- Pretrained models help in better generalization

---

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- WFDB

---

## License
MIT License

---

## Acknowledgements
- PhysioNet
- Open-source ML community
