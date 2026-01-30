# SeeFood — AI Image Classifier (Silicon Valley Remake)

A computer vision project inspired by the **SeeFood** app from *Silicon Valley*, which classifies images as **“Hotdog” or “Not Hotdog.”**  
The system uses a **Convolutional Neural Network (CNN)** trained on a Kaggle dataset and integrates **OpenCV** for image preprocessing and inference.

---

## 🎥 Project Inspiration

https://www.youtube.com/watch?v=tWwCK95X6go

---

## 🧠 Project Overview

This project recreates the core idea of the fictional *SeeFood* app by building a real, working image classifier. A CNN is trained to distinguish hotdogs from non-hotdogs using labeled image data, then applied to new images using an OpenCV-based pipeline.

The goal is to gain hands-on experience with the **end-to-end machine learning workflow**, including dataset handling, model training, and real-time image inference.

---

## 🧭 Approach

1. **Dataset Preparation**  
   A labeled image dataset is sourced from **Kaggle** and split into training and validation sets for supervised learning.

2. **CNN Training**  
   A Convolutional Neural Network is trained using using transfer learning to extract visual features and classify images as *hotdog* or *not hotdog*.

3. **Model Evaluation**  
   Model performance is evaluated using accuracy and validation loss to assess generalization.

4. **Inference Pipeline**  
   The trained model is used to classify new images, with OpenCV handling image loading and preprocessing prior to prediction.

---
