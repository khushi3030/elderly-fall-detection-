

# Fall Detection and Activity Recognition using Machine Learning and threshold based Methods

This repository contains the implementation and evaluation of distinct machine learning (ML) methods for addressing the challenges of fall detection and activity recognition. The project aims to develop effective and reliable systems to ensure the safety and well-being of individuals, particularly focusing on the elderly population.

## Methods

### 1. Machine learning Algorithms 

Applied Gaussian Naive Bayes,K-near neigbours,SVM,XGB classifier,Light GBM classifier, Multi Layer Perceptron  algorithms to UP-fall Dataset to compare accuracy of various techniques and find the most appropraite one and also to classify into fall and not fall.

### 2. Convolutional Neural Networks (CNNs)

The first method leverages convolutional neural networks (CNNs) to analyze sensor data and identify patterns indicative of falls or specific activities. CNNs are known for their ability to automatically learn hierarchical features from raw input data, making them well-suited for complex pattern recognition tasks.

### 3. Yolo(You Only Look Once)

The second method integrates the You Only Look Once (YOLO) framework with convolutional neural networks to achieve real-time object detection and activity recognition. YOLO is a state-of-the-art object detection system known for its speed and accuracy, making it ideal for applications where rapid processing of sensor data is crucial, such as fall detection scenarios.

### 4. Threshold Fusion

The third method employs threshold fusion, a technique that combines information from multiple sensor sources and applies predefined thresholds to determine the occurrence of falls or specific activities. By integrating data from different sensors and employing threshold-based decision-making, this method aims to enhance the robustness and reliability of fall detection and activity recognition systems.We calculate Theta and SVM values as threshold and then find optimum thrshold using roc curves and apply some algorithms to predict fall and not fall .


## Dataset

The dataset used in this project is UP-Fall and UR-Fall dataset . It comprises IMU sensor,accelorometer,infrared sensors  data crucial for training and testing the machine learning models. UR- Fall dataset compreises of images from Microsoft kinect cameras.

## Evaluation

Through comprehensive evaluation and comparative analysis, this project seeks to determine the strengths and limitations of each ML method in addressing fall detection and activity recognition challenges. The evaluation metrics include accuracy, precision, recall, and F1-score.

## Requirements

To run the code and reproduce the results, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Pandas
- CUDA 3.0
- GPU Drivers 
- NumPy
- Matplotlib

## Performance evaluation

A comprehensive set of performance metrics have been adopted to evaluate the effectiveness of different fall detection methods. These metrics include accuracy, precision, sensitivity, specificity, and F1-score, each providing valuable insights into the model's performance. Accuracy, calculated as the proportion of correctly classified instances among all instances, offers a broad overview of overall model performance. Precision, on the other hand, measures the proportion of true positive predictions among all positive predictions, providing insights into the model's ability to avoid false positives.
We applied various ML algorithms to the UP-Fall Dataset such as Decision Tree, Gaussian NB, XGB, LightGBM, K-means ,KNN ,SVM. Decision Forest turned out to be the best method providing high accuracy of 96 percent , precision and a greater no of false positives .The two implemented algorithms, THETA & SVM and SVM & THETA, were evaluated using our dataset and corresponding confusion matrices were generated. From the confusion matrices, performance metrics were calculated and ROC curves were plotted. ROC curves obtained for both algorithms by analyzing all threshold value pairs Theta & SVM algorithm  performed better .The accuracy obtained after applying fuzzy characteristic one is 54 percent and the accuracy obtained after applying fuzzy characteristics is 61 percent .The predicted labels represent the CNN's classification results, while the actual labels are the ground truth annotations for the images. This comparison is crucial for evaluating the model's performance. A grid of images has been displayed, where each image is annotated with both the predicted label and the actual label. This visual representation helps in identifying patterns in the model's predictions, such as which classes are being misclassified frequently or whether certain features are leading to incorrect predictions. The model has an accuracy of about 69.7 percent .Using the UR fall dataset, we enhanced real-time human fall detection accuracy with the Yolo method and its variants.These results support the fact that YOLOV8 is the best model for detecting falls in real-time.Yolov7 had the best accuracy about 94 per cent 



## Acknowledgments

- Special thanks to the creators and maintainers of the UP-Fall dataset.
- This project was inspired by the need for reliable fall detection and activity recognition systems to ensure the safety of individuals, especially the elderly population.
- The implementation of various Machine Learning and threshold based methods draws upon the collective knowledge and expertise of the machine learning community.
