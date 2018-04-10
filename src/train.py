#Author: Vineeth Chennapalli
#Big Data Genomics Assignment

import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from keras.models import Sequential 
from keras.optimizers import Adadelta
import matplotlib.pyplot as plt
import random

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

mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
train_tfbs, train_labels = get_train_data(TRAIN_PATH)

data = list(zip(train_tfbs, train_labels))
random.shuffle(data)
train_tfbs, train_labels = zip(*data)

tfbs_train, labels_train = np.array(train_tfbs), np.array(train_labels)

print("Creating LSTM Model . . .")

gcn = 1.25 #Gradient Clipping Norm
lstm_units = 32
batch_size = 20

embedding_matrix = np.zeros((4, 4))
embedding_matrix[0][0] = 1
embedding_matrix[1][1] = 1
embedding_matrix[2][2] = 1
embedding_matrix[3][3] = 1

optimizer = Adadelta(clipnorm = gcn)
model = Sequential()

model.add(Embedding(4, 4, weights = [embedding_matrix], input_length = 14, trainable = False))
model.add(Bidirectional(LSTM(units = lstm_units, return_sequences = True)))
model.add(Bidirectional(LSTM(units = lstm_units)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())
print("Starting training . . .")

#Training
model_history = model.fit(tfbs_train, labels_train, batch_size = batch_size, epochs = 15, validation_split = 0.2)
scores = model.evaluate(tfbs_train[1600:], labels_train[1600:], verbose = 0, batch_size = batch_size)
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
