import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

start_time = time.time()
# Define Training and Validation Data Generator with Augmentations
train_gen = ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=0.4,
    zoom_range=0.4
)
val_gen = ImageDataGenerator(
    rescale=1/255.,
)

# Flow the data into the Data Generator
Train = train_gen.flow_from_directory(
    "D:/TermPaper/TermPaperData/chest_xray/train",
    target_size=(224, 224),
    batch_size=16
)
Test = train_gen.flow_from_directory(
    "D:/TermPaper/TermPaperData/chest_xray/test",
    target_size=(224, 224),
    batch_size=8
)
print('Creating model')
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model and see it's structure and parameters
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print('Time before training', time.time() - start_time)
start_training_time = time.time()
# Training
print('Start training')
hist = model.fit_generator(
    Train,
    epochs=20,
    validation_data=Test
)

model.save("myCustomModel.hdf5")
end_time = time.time()
print( 'Time for all:',end_time - start_time)
print( 'Time for training:', end_time - start_training_time )
# Visualisation

plt.style.use("classic")
plt.figure(figsize=(16, 9))
plt.plot(hist.history['loss'], label="Train Loss")
plt.plot(hist.history['val_loss'], label="Valid Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over the Epochs")
plt.show()


plt.style.use("ggplot")
plt.figure(figsize=(16, 9))
plt.plot(hist.history['accuracy'], label="Train Accuracy")
plt.plot(hist.history['val_accuracy'], label="Valid Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy over the Epochs")
plt.show()
