# FUnIEGanAPI
## Developer - Ashiwin Rajendran

**FUnIE-GAN** (Fast Underwater Image Enhancement Generative Adversarial Network) is a conditional GAN-based model built to enhance underwater images by learning the mapping from distorted images (source domain X) to enhanced images (desired domain Y). The generator learns this mapping while competing against a discriminator in an adversarial framework. FUnIE-GAN is specifically optimized for **real-time inference**, making it ideal for applications like **Autonomous Underwater Vehicles (AUVs)** that require fast processing.

This directory contains the necessary scripts to run a **FastAPI** application that serves the FUnIE-GAN model. The API accepts blurry or distorted input videos of size 1920x1080 and returns enhanced, high-definition output videos.

### Key Features:
- **Real-time performance**: Optimized for fast inference to support real-time applications.
- **Small model size**: Designed to be deployed on actual AUVs for in-field use.
- **Flexible input size**: Supports any input video resolution as long as the dimensions are multiples of 16.
- **Faster inference**: Achieves efficient performance with minimal latency.
- **Cloud deployment**: Can be deployed on cloud platforms, allowing users to remotely upload videos for enhancement.
