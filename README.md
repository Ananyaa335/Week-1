Project title: Optimized Multi-Class Waste Classification (MobileNetV2)
# Week-1
This project successfully applies deep learning to the sustainability challenge of waste management. We developed a highly efficient Convolutional Neural Network (CNN) classifier capable of distinguishing between 12 distinct categories of garbage. The project emphasizes advanced optimization to achieve industry-standard performance and includes a practical solution for real-world clutter.


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




Week 3 -final submission:
 Core Technical Achievements

 1. Final Verified Accuracy
The model's performance was verified using synchronous internal Keras evaluation (`model.evaluate`).

Final Verified Accuracy:** **$\mathbf{92.12\%}$** (A strong result for fine-grained classification across 12 categories).
Base Model:** **MobileNetV2** (Chosen for its computational efficiency, making it ideal for deployment on resource-constrained hardware).

2. The Two-Stage Fine-Tuning Strategy

This advanced technique was critical to pushing the model's performance past the baseline.

Stage 1: Feature Extraction: =>Base MobileNetV2 layers were frozen. 
=>Quickly achieved $\sim 89.4\%$ accuracy by leveraging pre-trained general knowledge. 
Stage 2: Fine-Tuning: The top layers of MobileNetV2 were unfrozen and trained with a tiny learning rate ($\mathbf{1e-5}$). 
=>Specialized the features to recognize subtle differences in waste materials (e.g., distinguishing glass texture from plastic sheen), resulting in the final $\mathbf{92.12\%}$ accuracy. 

Solution to Real-World Limitations

Clutter & Occlusion: Sliding Window Pre-Processing (Implemented in the `app.py` frontend). The system analyzes the input image by cutting it into smaller, overlapping $\mathbf{224 \times 224}$ windows.
=> It then averages the predictions from windows with high confidence, guaranteeing clear classification even when items are overlapping. 
=>Deployment Readiness:Local CPU Inference. The final model loads and predicts accurately on the CPU, confirming it is ready for deployment without needing continuous, expensive cloud GPU resources. |

---

🚀 How to Run the Project Locally

1.  Clone the Repository:
    
    git clone [Your Repository URL]
2.  Install Dependencies:
    
    pip install tensorflow numpy gradio
    
3.  Execute the Application: Ensure model file (`final_optimized_waste_model.h5`) is present and run the app:   
    python app.py
4.  Open the local URL in your web browser to test the model's performance with cluttered images!
