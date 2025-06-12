from preprocessing import load_data
from model import build_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load dataset
x_train, x_test, y_train, y_test, label_map = load_data('HAM10000_Images/HAM10000_metadata.csv', 'HAM10000_Images/all_images')
# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2  # use 20% of training data for validation
)

# Create generators
train_generator = train_datagen.flow(x_train, y_train, batch_size=16, subset='training')
val_generator = train_datagen.flow(x_train, y_train, batch_size=16, subset='validation')


# Build model
model = build_model((64, 64, 3), len(label_map))

# Train
# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Train with generators
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)


# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save model
model.save('skin_lesion_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()
