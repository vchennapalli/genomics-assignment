#Author: Vineeth Chennapalli
#Big Data Genomics Assignment

import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential 
from keras.optimizers import Adadelta
import matplotlib.pyplot as plt

DATA_FOLDER = '../data'
TRAIN_PATH = DATA_FOLDER + '/train.csv'
TEST_PATH = DATA_FOLDER + '/test.csv'
MODEL_FOLDER = '../model'
SAVED_MODEL_PATH = MODEL_FOLDER + '/trained_model.json'
SAVED_WEIGHTS_PATH = MODEL_FOLDER + '/model_weights.h5'

def get_train_data(filepath):
    """
    input: filepath - string path to train data 
    output: train_tfbs, train_classes - tuple of lists of 14 character tfbs sequences and their corresponding classes
    """
    colnames = ['sequence', 'label']
    data = pd.read_csv(filepath, names = colnames)
    sequences, labels = data.sequence.tolist(), data.label.tolist()
    for i in range(1, 2001):
        seq = sequences[i]
        newseq = []
        for l in seq:
            newseq.append(mapping[l])
        sequences[i] = newseq
    return sequences[1:], labels[1:]

def get_test_data(filepath):
    """
    input: filepath - string path to test data
    output: test_tfbs - List of 14 character tfbs sequences 
    """
    colnames = ['sequence']
    data = pd.read_csv(filepath, names = colnames)
    sequences = data.sequence.tolist()
    for i in range(1, 401):
        seq = sequences[i]
        newseq = []
        for l in seq:
            newseq.append(mapping[l])
        sequences[i] = newseq
    return sequences[1:]


mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
train_tfbs, train_labels = get_train_data(TRAIN_PATH)
test_tfbs = get_test_data(TEST_PATH)
test_classes = []

#dividing the given set into training and validation sets in 4:1 ratio

tfbs_train, tfbs_validation = train_tfbs[:1600], train_tfbs[1600:]
labels_train, labels_validation = train_labels[:1600], train_labels[1600:]
#tfbs_train, tfbs_validation = np.array([list(word) for word in tfbs_train]), np.array([list(word) for word in tfbs_validation]) 
tfbs_train, tfbs_validation = np.array(tfbs_train), np.array(tfbs_validation)
labels_train, labels_validation = np.array(labels_train), np.array(labels_validation)
tfbs_train, tfbs_validation = tfbs_train.reshape(1600, 14, 1), tfbs_validation.reshape(400, 14, 1)
#labels_train, labels_validation = labels_train.reshape(1600, 1, 1), labels_validation.reshape(400, 2, 1)
print("Creating LSTM Model . . .")

gcn = 1.25 #Gradient Clipping Norm
lstm_units = 20
epochs = 1
batch_size = 50

optimizer = Adadelta(clipnorm = gcn)
model = Sequential()
#model.add(LSTM(lstm_units, dropout = 0.2, recurrent_dropout = 0.2))
model.add(LSTM(lstm_units, input_shape=(14, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())
print("Starting training . . .")

#Training

for epoch in range(epochs):
    model_history = model.fit(tfbs_train, labels_train, batch_size = batch_size, epochs = 10, validation_data = (tfbs_validation, labels_validation))
    print("Epoch: " + repr(epoch))

scores = model.evaluate(tfbs_validation, labels_validation, verbose = 0, batch_size = 1)
print("Accuracy: %.2f%%" % (scores[1]*100))

#saving the model
model_json = model.to_json()
with open(SAVED_MODEL_PATH, 'w') as jf:
     jf.write(model_json)
model.save_weights(SAVED_WEIGHTS_PATH)

plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show() 
