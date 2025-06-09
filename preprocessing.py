# import libraries
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import glob

def find_image_path(img_id, all_images_folder):
    # Search recursively inside all_images folder for the image filename b/c we've 2 folders inside
    pattern = os.path.join('HAM10000_Images/all_images', '**', img_id + '.jpg')
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None

IMAGE_SIZE = 64
def load_data(metadata_path, all_images_folder):
    df = pd.read_csv('HAM10000_Images/HAM10000_metadata.csv')
    label_map = {label: idx for idx, label in enumerate(df['dx'].unique())} #Creates a dictionary mapping each disease name (like nv, mel, etc.) to a unique integer(0-6)
    df['label'] = df['dx'].map(label_map)

    # creating list to store the image data and labels
    images = []
    valid_labels = []

    for _, row in df.iterrows():
        img_id = row['image_id']
        img_path = find_image_path(img_id, all_images_folder)

        if img_path is None:
            print(f"[WARNING] Image not found: {img_id}.jpg")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read image: {img_path}")
            continue

        img = cv2.resize(img, (64, 64))
        images.append(img)
        valid_labels.append(row['label']) #Adds the processed image and its label to your datasets.

    images = np.array(images)
    labels = to_categorical(valid_labels, num_classes=len(label_map)) # Converts integer labels to vectors

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)

    return x_train, x_test, y_train, y_test, label_map

