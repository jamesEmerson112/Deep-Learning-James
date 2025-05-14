# Level 1: Basic PyTorch Image Classifier

## 🧠 Project Overview
Build a simple CNN classifier on the CIFAR-10 dataset using PyTorch. This project introduces the fundamentals of deep learning and PyTorch's model training pipeline.

### 🧒 Simple Explanation
The CIFAR-10 dataset contains 60,000 small images (32x32 pixels), and each image shows one object like a cat, car, or airplane. There are 10 different categories total. Our goal is to teach a computer program (called a CNN) to look at a new image and correctly guess which of the 10 categories it belongs to. It’s like training a mini brain to recognize what’s in a picture!

---

## 🚀 Using NVIDIA NGC for Deep Learning

NVIDIA NGC (NVIDIA GPU Cloud) is a platform that provides GPU-optimized software, containers, pre-trained models, and resources for AI and deep learning. For this project, you can use NGC to access official PyTorch containers and models, ensuring maximum performance and compatibility with NVIDIA GPUs.

### How to Use NGC for This Project

1. **Sign Up and Authenticate**
   - Register for an NVIDIA Cloud Account and create an NGC Org.
   - Generate an NGC API key for authenticated access.

2. **Choose Your Platform**
   - You can run NGC containers on your own NVIDIA GPU machine, on NVIDIA DGX systems, or on cloud platforms (AWS, Azure, GCP, Alibaba) using NVIDIA GPU-optimized VM images.

3. **Pull and Run Containers**
   - Use Docker (with NVIDIA Container Toolkit) to pull and run containers from the NGC registry (`nvcr.io`).
   - Example:
     ```
     docker login nvcr.io --username '$oauthtoken' --password <NGC_API_KEY>
     docker pull nvcr.io/nvidia/pytorch:<tag>
     docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:<tag>
     ```
   - Mount your code and data into the container using Docker volume flags.

4. **Use Pre-trained Models and Resources**
   - Download models and resources using the NGC CLI or web UI.
   - Fine-tune or use them directly in your research.

### Cloud GPU with NVIDIA Services

- NGC provides the software and containers, but not direct GPU compute. To use cloud GPUs, launch a VM with an NVIDIA GPU on a major cloud provider and use NGC containers there.
- NVIDIA publishes ready-to-use VM images on AWS, Azure, GCP, and Alibaba Cloud, pre-installed with drivers and the NVIDIA Container Toolkit.

**Reference:** See the included `ngc-user-guide.pdf` for detailed instructions and best practices.

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

### Tech Stack

- **Programming Language:** Python 3.x
- **Deep Learning Framework:** PyTorch (with torchvision)
- **Data Handling:** torchvision.datasets.CIFAR10, torch.utils.data.DataLoader
- **Model Development:** torch.nn.Module, torch.nn.functional
- **Training Utilities:** torch.optim (Adam/SGD), torch.nn.CrossEntropyLoss
- **Experiment Tracking (Optional):** TensorBoard, Weights & Biases, or MLflow
- **Environment:** NVIDIA NGC PyTorch Docker Container (with GPU support), requirements.txt (if running outside Docker)
- **Hardware:** NVIDIA GPU (local or cloud), CPU fallback
- **Cloud Integration:** NVIDIA NGC for containers/models, Cloud VM (AWS, Azure, GCP, Alibaba) for GPU resources

#### Tech Stack Flow

```
[User Code: Python]
      |
      v
[PyTorch Framework]
      |
      v
[NGC PyTorch Docker Container]
      |
      v
[NVIDIA GPU Drivers & CUDA Toolkit]
      |
      v
[Hardware: GPU/CPU (Local or Cloud VM)]
```

### High-Level Flow (ASCII Diagram)

```
+---------------------+
| Download CIFAR-10   |
+---------------------+
           |
           v
+---------------------+
| Preprocess &        |
| Augment Images      |
+---------------------+
           |
           v
+---------------------+
| Load Data in Batches|
+---------------------+
           |
           v
+---------------------+
|   Feed to CNN Model |
+---------------------+
           |
           v
+---------------------+
| Compute Loss &      |
| Backpropagation     |
+---------------------+
           |
           v
+---------------------+
| Update Model Weights|
+---------------------+
           |
           v
+---------------------+
| Evaluate on         |
| Validation Set      |
+---------------------+
           |
           v
+---------------------+
|   Converged?        |
+---------------------+
     |         |
   No|         |Yes
     v         v
+---------------------+      +----------------------+
|   Repeat Training   |      | Save Model &         |
|   (Next Epoch)      |      | Run Inference        |
+---------------------+      +----------------------+
```

### Architecture
- Input: RGB images of pet faces (various sizes, resized as needed)
- Model: Convolutional Neural Network (CNN)
  - Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Dropout → FC → Softmax
- Output: Emotion classification (e.g., Angry, Happy, Sad, etc.)

### Components
- DataLoader: Loads pet facial expression images from class-named folders (e.g., Angry, Happy, Sad)
- Model: Custom CNN
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Metrics: Accuracy

### Design Decisions
- Use dropout to prevent overfitting
- Normalize images for consistent input
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
