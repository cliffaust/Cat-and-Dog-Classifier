# import some of the librabries for processing the image and training the algorithm
import cv2 
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

#process the data
#the data is collected from kaggle 
# you can download it using the this link https://www.kaggle.com/tongpython/cat-and-dog
train_data = []
test_data = []
data_list = ['cats', 'dogs']
size = 32

#the data is converted to an array and resize
# an exception is used because some of the images are broken
def train():
    for data in data_list:
  	# we use os to join the images together
        path = os.path.join('training_set/training_set', data)
    #each of the data in data_list is converted to index, that is 0 and 1
        classes = data_list.index(data)
        for image in os.listdir(path):
            try:
      	#open cv is used for the convertion of the images
                image_array = cv2.imread(os.path.join(path, image))
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(image_array, (size,size))
        # the converted images are appended to a list
                train_data.append([new_array, classes])
            except:
                pass

# we apply thesame principle as in he train function
def test():
    for data in data_list:
        path = os.path.join('test_set/test_set', data)
        classes = data_list.index(data)
        for image in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path, image))
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(image_array, (size, size))
                test_data.append([new_array, classes])
            except:
                pass

training()
test()

random.shuffle(train_data)
random.shuffle(test_data)

x_train = []
y_train = []
x_test = []
y_test = []

for features, labels in train_data:
	# the features are appended to the x_train
    x_train.append(features)
	# the labels are appended to the y_train
    y_train.append(labels)

for features, labels in test_data:
	# the features are appended to the x_test
    x_test.append(features)
    # the features are appended to the y_test
    y_test.append(labels)
  
#the images is then converted to a numpy array(as required by keras) and reshape
x_train = np.array(x_train).reshape(-1, size, size, 3)
x_test = np.array(x_test).reshape(-1, size, size, 3)

#the features and labels are then saved in a file
print('Saving the dataset -----')
np.save('x_train', x_train)
np.save('x_test', x_test)

np.save('y_train', y_train)
np.save('y_test', y_test)
print('Done---')

num_classes = 1
epochs = 30
batch_size = 32
channels = 3

# The image value is required to be in between 0 and 1(as it will be easy for the compter to compute)
x_train = x_train.astype('float32')
x_train /= 225

x_test = x_test.astype('float32')
x_test /= 225

# we use the sequential model
model = Sequential()

# the network architecture is created
model.add(Conv2D(32, (3,3), input_shape=(size,size,channels), kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

# I used a sigmoid activation function because i want the value to be in probability format
model.add(Dense(num_classes, activation='sigmoid'))

# the model is compiled using loss function of binary(thus simply: yes or no)
model.compile(optimizer='adam', 
             loss='binary_crossentropy',
             metrics=['accuracy'])

print('running model ---')
hist = model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size)
print('Done ---')

#the model is tested on the test set
print('testing model ---')
model.evaluate(x_test, y_test)
print('Done --')

#The behaviour of the network is checked by printing a graph
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend(['train','valid'])
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Loss')
plt.legend(['train','valid'])
plt.show()

# The model if tested further using a random image from any source
# Here i used a cat image from my computer to make the test
# You can using yours
image = cv2.imread('cat.18.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (size,size))
image = np.array(image).reshape(-1, size, size, 3)

prediction = model.predict(image)
if prediction >= 0.5:
  print('Dog')
else:
  print('Cat')
