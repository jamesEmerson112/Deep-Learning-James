# Level 1: Basic PyTorch Image Classifier

## 🧠 Project Overview
Build a simple CNN classifier on the CIFAR-10 dataset using PyTorch. This project introduces the fundamentals of deep learning and PyTorch's model training pipeline.

### 🧒 Simple Explanation
The CIFAR-10 dataset contains 60,000 small images (32x32 pixels), and each image shows one object like a cat, car, or airplane. There are 10 different categories total. Our goal is to teach a computer program (called a CNN) to look at a new image and correctly guess which of the 10 categories it belongs to. It’s like training a mini brain to recognize what’s in a picture!

---

## 🔍 Input, Output, Edge Cases, and Constraints

### ✅ Input
- A single 32x32 color image (RGB) from the CIFAR-10 dataset.
- Each image contains one object from one of 10 categories (e.g., airplane, cat, truck).

### 🎯 Output
- A predicted label (0–9) corresponding to one of the 10 categories.
- Optionally, a list of probabilities for each class (softmax output).

### ⚠️ Edge Cases
- Image is corrupted or unreadable → should raise an error or skip.
- Image is not 32x32 or not RGB → should be resized or converted.
- Empty input or batch → should return a warning or no prediction.

### 📌 Constraints
- Must work on both CPU and GPU (if available).
- Should train within a reasonable time (e.g., under 30 minutes on GPU).
- Keep model size small enough to run on limited hardware (e.g., laptops).

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
