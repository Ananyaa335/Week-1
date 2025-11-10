Project title: Optimized Multi-Class Waste Classification (MobileNetV2)
# Week-1
This project uses a Convolutional Neural Network (CNN) to automatically identify and classify images of waste into 12 different material categories (e.g., Plastic, Metal, Paper, E-waste, Organic).  The key is implementing Transfer Learning and Fine-Tuning on a pre-trained model.

Sustainability Relevance:
Problem Addressed: Inefficient waste sorting leads to high landfill contamination, increased processing costs, and lower recycling rates. Solution: By automating the classification process, the system can dramatically improve the purity of recycling streams, making the recycling process more economically viable and environmentally effective. The focus on detailed classes like differentiating battery waste or PCB e-waste.Thus it addresses modern, complex waste challenges.

Week 2 Progress Summary: Model Training

Project Overview:
This project applies deep learning techniques to solve the sustainability challenge of efficient waste sorting. We train a Convolutional Neural Network (CNN) to classify images of household garbage into 12 distinct categories, aiming for the high accuracy required for automated recycling systems.

Key Technologies Used
* Framework: TensorFlow / Keras (Python)
* Model: MobileNetV2 (Pre-trained on ImageNet)
* Data: Garbage Classification (12 classes) - Kaggle
* Hardware: Google Colab GPU (NVIDIA T4)

In Week 2, I successfully implemented the core of the Transfer Learning process:
Steps done :
1. Data Preparation and Augmentation (Phase 1)
* Successfully downloaded, unzipped, and loaded 15,515 images across 12 classes.
* Applied Data Augmentation (rotation, flip, zoom, shift) to the training data to prevent overfitting and improve generalization.

2. Feature Extraction (Phase 2 - Initial Training)
   Model Setup: Loaded the MobileNetV2 backbone and froze all of its pre-trained layers. We only trained a small, custom classification head for 10 epochs.
Results: This phase quickly established a strong baseline:
    * Peak Training Accuracy: ~93%
    * Peak Validation Accuracy (Baseline): ~89.4%
Status: This confirmed that the MobileNetV2 features are highly effective for identifying waste types.

4. Fine-Tuning Preparation (Phase 3 - Ready to Run)
* The model is now configured for the next advanced step: Fine-Tuning.
* We have unfrozen the last 30 layers of the MobileNetV2 backbone and set the learning rate extremely low.
* The code is ready to train to maximize accuracy by optimizing the deeper feature layers on the waste dataset.

---

Next Step (Fine-Tuning):
*To Run the Fine-Tuning phase to achieve $\approx 95\%$+ validation accuracy and analyze the performance per class using the Classification Report.
