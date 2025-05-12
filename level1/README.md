# Level 1: Basic PyTorch Image Classifier

## 🧠 Project Overview
Build a simple CNN classifier on the CIFAR-10 dataset using PyTorch. This project introduces the fundamentals of deep learning and PyTorch's model training pipeline.

## 🧱 System Design

### Architecture
- Input: 32x32 RGB images
- Model: Convolutional Neural Network (CNN)
  - Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Dropout → FC → Softmax
- Output: 10-class classification

### Components
- DataLoader: CIFAR-10 dataset with transforms
- Model: Custom CNN
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Metrics: Accuracy

### Design Decisions
- Use dropout to prevent overfitting
- Normalize CIFAR-10 images
- Use GPU if available

## 💬 Prompt Engineering Notes

Use GPT-4o to assist with:
- Debugging training issues
- Designing CNN architecture
- Understanding PyTorch APIs

**Example Prompts:**
- "Provide a simple CNN PyTorch template with dropout layers."
- "Debug: CNN accuracy stuck at 50%."

## 🔬 Experimentation & Results

| Experiment | Description | Accuracy | Notes |
|------------|-------------|----------|-------|
| Exp1       | Baseline CNN |          |       |
| Exp2       | Added dropout |         |       |
| Exp3       | Data augmentation |     |       |

## 📚 References & Resources
- [PyTorch CIFAR-10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [CNN Architectures](https://cs231n.github.io/convolutional-networks/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
