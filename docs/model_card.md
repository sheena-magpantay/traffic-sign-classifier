# MODEL CARD-TRAFFIC SIGN CLASSIFIER


Model Details
----------------------------------------------
 - A Traffic Sign Classification model based on a Convolutional Neural Network (CNN) architecture.

- Designed to classify road and traffic signs from image inputs.

- Built with image preprocessing powered by OpenCV.

- Developed as an academic project.

Intended Use
----------------------------------------------
This model is developed strictly for academic and educational purposes.

Factors
----------------------------------------------

  - Performance may vary depending on:

   - Lighting conditions (day/night, shadows)
     
   - Image quality and resolution


Dataset
----------------------------------------------

Dataset: Philippine Traffic Sign Dataset (Roboflow Universe) by Jerry
License: CC BY 4.0
Size: 5,895 images
Official run subset: Approximately 70% training, 20% validation, and 10% testing split 
Cleaning: Images were manually annotated and reviewed; duplicate, and incorrectly labeled samples were minimized during preprocessing
Splits: Dataset was split into training, validation, and testing sets using Roboflow’s automated split to ensure balanced class distribution

Metrics
----------------------------------------------

Classification metrics:
  - Accuracy, Precision, Recall, F1-score
    
  - RL Cumulative Reward

Performance (Initial Results)
----------------------------------------------

  - CNN Model: 70% accuracy

  - Training shows decreasing loss and improving accuracy over epochs

Ethical Considerations
----------------------------------------------

  - Evaluated in a controlled simulation environment only.

  - Intended strictly for academic and experimental use.
