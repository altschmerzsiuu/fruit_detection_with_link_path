# Fruit and Vegetable Classifier
This project implements a fruit and vegetable classification system using a Convolutional Neural Network (CNN) based on MobileNetV2. The system uses transfer learning to classify images into 10 categories, leveraging a pre-trained model on ImageNet for feature extraction.

## Features
- Data Augmentation: Enhances the training data with transformations like rotation, zoom, and flipping.
- MobileNetV2 Transfer Learning: Uses a lightweight pre-trained model to improve performance with limited training data.
- Training Visualization: Includes plots for training and validation accuracy/loss.
- Custom Prediction Functionality: Accepts input images to predict the corresponding fruit or vegetable.
- Interactive Command Line Interface: Allows continuous prediction for user-provided image paths.

## Installation
### Prerequisites
-Ensure you have the following installed:
    - Python 3.8 or above
    - TensorFlow 2.x
    - NumPy
    - Matplotlib
- Install the required Python libraries:
  ```bash
      pip install tensorflow numpy matplotlib

## Dataset Structure
- Organize the dataset into the following folder structure:
  ```bash
  Data Science Projects/
  └── Fruit_Vegie/
      ├── train/
      │   ├── apple/
      │   ├── banana/
      │   ├── durian/
      │   └── ... (other categories)
      └── test/
          ├── apple/
          ├── banana/
          ├── durian/
          └── ... (other categories)
- train/: Contains training images, divided into folders by category.
- test/: Contains testing images, divided into folders by category.

## Usage
### Training the Model
1. Place your dataset in the appropriate structure under train/ and test/ directories.
2. Run the script to train the model. The model will be saved as fruit_classifier_model.h5.
    ```bash
    python fruit_classifier.py

### Testing the Model
The script includes an interactive CLI for image predictions. Provide the file path to an image, and the model will classify it.

1. Start the prediction loop:
   - ```bash
     python fruit_classifier.py
2. Enter the image path when prompted:
   - ```bash
     Enter the file path of the image (or type 'x' to quit): path/to/image.jpg
3. View the classification result.


## Outputs
- Training Accuracy/Loss Plots: Visualize the model's learning progress over epochs.
- Test Accuracy: Evaluate the model on unseen test data.
- Predictions: Classify user-provided images.

## Labels
The model is trained to classify the following categories:

1. Apple
2. Banana
3. Durian
4. Jackfruit
5. Mango
6. Orange
7. Pineapple
8. Pomegranate
9. Tomato
10. Watermelon

## Example Usage
1. Train the model:
   - ```bash
     python fruit_classifier.py
2. Predict an image:
   - ```bash
     Enter the file path of the image (or type 'x' to quit): path/to/image.jpg
     The image matches with: Mango

## Notes
- Ensure the dataset is preprocessed (cropped and labeled) before training.
- You can adjust the number of categories and labels based on your dataset.

## Acknowledgments
- Pretrained MobileNetV2 model from TensorFlow/Keras.
- Dataset should be collected and organized manually or sourced from an open dataset platform.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
