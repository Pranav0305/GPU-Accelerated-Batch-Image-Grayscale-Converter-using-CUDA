# GPU-Accelerated Batch Image Grayscale Converter using CUDA

## 1. Project Overview
This project demonstrates the use of **GPU computing with CUDA** to accelerate image processing tasks. The application converts RGB images to grayscale by leveraging **massive data parallelism** on the GPU. Each CUDA thread processes one pixel, making this program highly scalable and well-suited for large images or batches of images.

The goal of this project is to apply GPU programming concepts learned in this course—such as kernel execution, memory management, and parallel computation—to a real-world and practical workload.

---

## 2. Motivation
Image processing is a common and computationally expensive task in fields such as computer vision, medical imaging, robotics, and multimedia systems. GPUs are particularly effective for such workloads because the same operation is applied independently to millions of pixels.

This project aligns with my personal learning goals of:
- Understanding CUDA kernel design
- Learning GPU memory allocation and data transfers
- Applying GPU parallelism to a real-world problem

---

## 3. Technologies Used
- **CUDA C++**
- **NVIDIA GPU**
- **CUDA Runtime API**
- **NVCC Compiler**

No CPU-only or multithreaded implementations are used. All computation is performed on the GPU.

---

## 4. System Requirements
- NVIDIA GPU with CUDA support  
- CUDA Toolkit installed (CUDA 11 or higher recommended)  
- Linux or Windows system with NVCC available  

To verify CUDA installation:
```bash
nvcc --version
