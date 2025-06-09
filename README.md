# Skin_lesion_Classifier
This project is a deep learning-based skin lesion classification system built using TensorFlow, Keras, and Streamlit. It is trained on the HAM10000 dataset, which consists of dermatoscopic images of common pigmented skin lesions. The model uses Convolutional Neural Networks (CNNs) to classify lesions into 7 categories and provides predictions along with confidence scores through a simple web interface.
Project Overview
Dataset: HAM10000 (Human Against Machine with 10000 training images)

Model: CNN trained from scratch

Technologies:

Python

TensorFlow / Keras

OpenCV

Pandas, NumPy

Streamlit (for GUI)

How It Works
Preprocessing: Images are read, resized (64x64), and normalized. Labels are encoded using to_categorical.

Model Training: CNN model is trained using model.fit() or optionally ImageDataGenerator for data augmentation.

Prediction: A Streamlit GUI allows users to upload an image and receive:
Preview of the uploaded image

Predicted class name

Confidence score



Classes of Skin Lesions

0	Melanocytic nevi (nv)

1	Melanoma (mel)

2	Benign keratosis-like lesions (bkl)

3	Basal cell carcinoma (bcc)

4	Actinic keratoses (akiec)

5	Vascular lesions (vasc)

6	Dermatofibroma (df)

