# Extract features
import os, shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.models import Model

datagen = ImageDataGenerator(rescale=1./255) #Rescale since data ranges between 0 and 255.
batch_size = 32
img_width = 224
img_height = 224

# imports the mobilenet model and discards the last 1000 neuron layer.
base_model = MobileNet(weights='imagenet', 
                       include_top=False, 
                       input_shape=(img_height, img_width, 3),
                       pooling='avg')

# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name, layer.trainable)
#     print(layer.output_shape)

train_dir = 'data/raw/train'
validation_dir = 'data/raw/validation'
test_dir = 'data/raw/test'

def extract_features(directory, sample_count):

    classes_count = len(os.listdir(directory))
    # Must be equal to the output of the convolutional base
    features = np.zeros(shape=(sample_count, 1024))
    labels = np.zeros(shape=(sample_count, classes_count))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode='categorical')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        print('Batch Input Shape =', inputs_batch.shape)
        print('Batch Label Shape =', labels_batch.shape)
        features_batch = base_model.predict(inputs_batch)
        print('Batch Predict Feature Shape =', features_batch.shape)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        print('features from {} to {}'.format(i*batch_size, (i+1)*batch_size))
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

def get_sample_size(directory):
    size = 0
    for _, d, _ in os.walk(directory):
        for dir in d:
            class_dir = os.path.join(directory, dir)
            for _, _, f in os.walk(class_dir):
                size += len([file for file in f if '.jpg' in file])
    return size

train_size = get_sample_size(train_dir)
validation_size = get_sample_size(validation_dir)
test_size = get_sample_size(test_dir)

train_features, train_labels = extract_features(train_dir, train_size)
validation_features, validation_labels = extract_features(validation_dir, validation_size)
test_features, test_labels = extract_features(test_dir, test_size)

feat_directory = 'data/processed/'

np.save(feat_directory+'train_features', train_features)
np.save(feat_directory+'train_labels', train_labels)
np.save(feat_directory+'validation_features', validation_features)
np.save(feat_directory+'validation_labels', validation_labels)
np.save(feat_directory+'test_features', test_features)
np.save(feat_directory+'test_labels', test_labels)