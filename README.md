# Deep Learning-Based Detection and Severity Prediction of Dementia Using MRI Scans

## Project Overview
This project applies deep learning techniques to detect and predict the severity of dementia using MRI scans, enhancing interpretability with Grad-CAM visualizations. We employ convolutional neural networks (CNN) and Gradient-weighted Class Activation Mapping (Grad-CAM) to analyze region-specific brain activity and detect various stages of dementia.

## Objectives
- **Detection**: Classify the severity of dementia into four classes using MRI scans.
- **Interpretability**: Employ Grad-CAM to highlight significant brain regions influencing the predictions, providing visual explanations for the model's decisions.

## Dataset
- **Source**: Kaggle
- **Initial Size**: 6,400 images across four classes with significant class imbalance.
- **Augmented Size**: Increased to 6,400 balanced images using Deep Convolutional GANs for more robust training.

## Methodology
- **CNN Models**: Tested various architectures like InceptionV3, DenseNet, ResNet, and EfficientNet.
- **Data Augmentation**: Utilized DCGAN to balance the class distribution in the training dataset.
- **Grad-CAM Visualization**: Integrated Grad-CAM to generate heatmaps that illustrate the areas of the brain most relevant to the model’s predictions.

## Results
- **Best Model**: CNN with an accuracy of 87%.
- **Performance Metrics**:
  - Precision (Macro Avg): 0.87
  - Recall (Macro Avg): 0.87
  - F1-Score (Macro Avg): 0.87
  - Accuracy: 0.87

## Training and Validation
- Displayed consistent improvement over epochs, with CNN and DenseNet performing best on unseen data.
- Demonstrated balanced learning with minimal overfitting or underfitting, confirmed by close training and validation accuracy.

## Grad-CAM Results
Grad-CAM visualizations provided insights into which brain regions are most affected in different stages of dementia, aiding in the clinical analysis and understanding of the disease.

## Challenges
- **Data Imbalance**: Addressed using DCGAN which was computationally intensive and initially hindered by hardware limitations.
- **Model Training**: Optimization of deep learning models required significant computational resources, which were mitigated by enhanced GPU support.

## Achievements
- Achieved high accuracy in dementia classification.
- Successfully implemented interpretability with Grad-CAM, enhancing trust in the model's diagnostic recommendations.

## Future Work
- Explore additional augmentation techniques like StyleGAN.
- Incorporate multimodal data to enrich model predictions.
- Expand the model's application to other neurodegenerative diseases.

## Citations
Refer to the papers and resources that influenced this project:
- [Science Direct Article on Deep Learning in Medicine](https://www.sciencedirect.com/science/article/pii/S1110016822005191)
- [Nature Article on Neural Networks](https://www.nature.com/articles/s41467-022-31037-5)
- [IEEE Paper on CNN Applications](https://ieeexplore.ieee.org/document/9587953)



