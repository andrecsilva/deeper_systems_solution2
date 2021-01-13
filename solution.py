import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

input_shape = (64,64,3)

csv_file = open('train.truth.csv')
#skip first line
csv_file.readline()

csv_reader = csv.reader(csv_file,delimiter=',')
train = [(row[0],row[1]) for row in csv_reader]

files = [image for (image,label) in train]
labels = [label for (image,label) in train]

#translate labels to numbers
label_translator = dict([(b,a) for (a,b) in list(enumerate(set(labels)))])
labels = [label_translator[label] for label in labels]

#tensorize the data and put into a database
images = tf.constant(files)
labels = tf.constant(labels)
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

def parse_function(filename,label):
    image_string = tf.io.read_file('train/'+filename)
    image_decoded = tf.image.decode_image(image_string)
    image = tf.cast(image_decoded,tf.float32)
    return image,label
    
#translates the dataset with filenames with actual images
dataset = dataset.map(parse_function).batch(len(files))
i = iter(dataset)
dataset = iter(dataset).get_next()

train_images = dataset[0]
train_labels = dataset[1]

#normalize
train_images = train_images / 255

print(train_images.shape)
print(train_labels.shape)

from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D

#model
model = tf.keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='relu'))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

#load test data
test_files = os.listdir('test')
test_images = tf.constant(test_files)
test_dataset = tf.data.Dataset.from_tensor_slices(test_files)

def map_test(filename):
    image_string= tf.io.read_file('test/'+filename)
    image_decoded = tf.image.decode_image(image_string)
    image = tf.cast(image_decoded,tf.float32)
    return image

test_dataset = test_dataset.map(map_test).batch(len(test_files))
test_dataset = iter(test_dataset).get_next()

test_dataset = test_dataset/255

#predict
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
prediction = probability_model(test_dataset)
predictions = [np.argmax(p) for p in prediction]
labelled_predictions = [list(label_translator)[l] for l in predictions]

#write csv with predictions
csv_data = list(zip(test_files,labelled_predictions))
csv_file = open('pred.csv',mode='w')
csv_writer = csv.writer(csv_file,delimiter=',')
csv_writer.writerows(csv_data)

label_rotations = dict(zip(list(label_translator),[180,270,0,90]))

from PIL import Image

for file,label in csv_data:
    image = Image.open('test/'+file)
    image = image.rotate(label_rotations[label])
    image.save('test_rotated/'+file)
