# Importing required libraries
import os  # For handling file and directory operations
import numpy as np  # For numerical operations, especially array manipulation
import matplotlib.pyplot as plt  # For plotting graphs (accuracy and loss)
import tensorflow as tf  # For building and training the neural network
from tensorflow.keras.models import Sequential  # For creating a sequential neural network
from tensorflow.keras.layers import Dense, Flatten, Dropout  # Layers for the neural network
from tensorflow.keras.applications import MobileNetV2  # Pretrained MobileNetV2 for transfer learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  # Image preprocessing

# Data augmentation for training dataset
# This will apply transformations like rotation, flipping, and zooming to augment the training data.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,           # Normalize pixel values to [0, 1] range
    rotation_range=20,           # Randomly rotate images up to 20 degrees
    width_shift_range=0.2,       # Randomly shift images horizontally by 20%
    height_shift_range=0.2,      # Randomly shift images vertically by 20%
    shear_range=0.2,             # Apply shearing transformations
    zoom_range=0.2,              # Randomly zoom in/out
    horizontal_flip=True,        # Randomly flip images horizontally
)

# Only rescale pixel values for the test dataset (no augmentation for testing)
# The test set is used to evaluate the model and should not be augmented
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load and preprocess the training data
# The training dataset is loaded from the 'train' directory. The images are resized to (224, 224).
train_data = train_datagen.flow_from_directory(
    r"...\Data Science Projects\Fruit_Vegie\train",  # Training dataset path
    target_size=(224, 224),  # Resize all images to 224x224 (required by MobileNetV2)
    batch_size=32,           # Number of images per batch
    class_mode='categorical',  # Multi-class classification (one-hot encoded labels)
    color_mode='rgb'         # Use RGB images (3 color channels)
)

# Load and preprocess the test data
# Similarly, the test dataset is loaded from the 'test' directory and resized to (224, 224).
test_data = test_datagen.flow_from_directory(
    r"...\Data Science Projects\Fruit_Vegie\test",  # Testing dataset path
    target_size=(224, 224),  # Resize all images to 224x224
    batch_size=32,           # Number of images per batch
    class_mode='categorical',  # Multi-class classification
    color_mode='rgb'         # Use RGB images
)

# Load the MobileNetV2 model with pretrained ImageNet weights
# We are using MobileNetV2, which has been pretrained on ImageNet, as the base model for transfer learning.
# We exclude the top layers and add our own custom layers for classification.
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base layers (no training for pretrained layers)

# Build the custom model using the pretrained MobileNetV2 as a base
# We add custom layers on top of the base MobileNetV2 model for our specific task.
model = Sequential([
    base_model,  # Add the pretrained MobileNetV2 as the base
    Flatten(),  # Flatten the feature map to a 1D vector
    Dropout(0.25),  # Add dropout to prevent overfitting during training
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons and ReLU activation
    Dense(10, activation='softmax')  # Output layer for 10 classes (softmax for probabilities)
])

# Compile the model
# Here, we use the Adam optimizer, categorical crossentropy loss (since it's a multi-class classification task), 
# and accuracy as the metric for evaluation.
model.compile(
    optimizer='adam',  # Adam optimizer for adaptive learning
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

# Train the model using the training data and validate on the test data
# The model is trained for 15 epochs. After each epoch, it is evaluated on the test dataset to check performance.
history = model.fit(
    train_data,          # Training data
    epochs=15,           # Number of epochs to train
    validation_data=test_data,  # Validation data
)

# Plot training and validation accuracy over epochs
# The following plots show how the training and validation accuracy evolve during the training process.
plt.figure(figsize=(5, 5))
plt.plot(history.history['accuracy'], color='red', label='Training Accuracy')  # Training accuracy
plt.plot(history.history['val_accuracy'], color='blue', label='Validation Accuracy')  # Validation accuracy
plt.xlabel('Epochs')  # Label for x-axis
plt.ylabel('Accuracy')  # Label for y-axis
plt.title('Training and Validation Accuracy')  # Graph title
plt.legend()  # Show legend
plt.show()  # Show the plot

# Plot training and validation loss over epochs
# Similarly, we plot the loss curves to see how the loss improves during training.
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], color='red', label='Training Loss')  # Training loss
plt.plot(history.history['val_loss'], color='blue', label='Validation Loss')  # Validation loss
plt.xlabel('Epochs')  # Label for x-axis
plt.ylabel('Loss')  # Label for y-axis
plt.title('Training and Validation Loss')  # Graph title
plt.legend()  # Show legend
plt.show()  # Show the plot

# Evaluate the model on the test dataset
# The model's performance on the test dataset is evaluated here.
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.2f}")  # Print the test accuracy

# Save the trained model to a file
# After training, the model is saved to a file 'fruit_classifier_model.h5' for later use.
model.save('fruit_classifier_model.h5')  # Save the model as a `.h5` file

# Define fruit labels (ensure these match the directory class indices)
# These labels correspond to the folder names in the 'train' and 'test' directories.
labels = ['apple', 'banana', 'durian', 'jackfruit', 'mango', 
          'orange', 'pineapple', 'pomegranate', 'tomato', 'watermelon']

# Function to predict the class of a fruit from an image file
# This function accepts an image file path, preprocesses it, and predicts the fruit class using the trained model.
def predict_fruit(image_path):
    # Load and preprocess the input image
    img = load_img(image_path, target_size=(224, 224))  # Resize to 224x224
    img = img_to_array(img) / 255.0  # Convert to array and normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension for prediction

    # Make predictions using the trained model
    predictions = model.predict(img)
    class_index = np.argmax(predictions)  # Get the index of the highest probability
    label = labels[class_index]  # Map the index to the corresponding label

    print(f"The image matches with: {label}")  # Print the predicted label

# Print the class indices to verify label mappings
print(train_data.class_indices)

# Loop for continuous image predictions
# This section allows the user to continuously input image paths and predict the class of the fruit.
while True:
    image_path = input("Enter the file path of the image (or type 'x' to quit): ")  # Input image path
    if image_path.lower() == 'x':  # Exit the loop if user types 'x'
        print("Exiting the program.")
        break
    else:
        predict_fruit(image_path)  # Predict the class of the input image
