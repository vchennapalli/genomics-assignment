#Author: Vineeth Chennapalli
#Big Data Genomics Assignment
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential 
from keras.optimizers import Adadelta
from keras.models import model_from_json

DATA_FOLDER = '../data'
TEST_PATH = DATA_FOLDER + '/test.csv'
MODEL_FOLDER = '../model'
SAVED_MODEL_PATH = MODEL_FOLDER + '/trained_model.json'
SAVED_WEIGHTS_PATH = MODEL_FOLDER + '/model_weights.h5'
RESULTS = '../results/predictions.csv'

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
test_tfbs = get_test_data(TEST_PATH)

tfbs_test = np.array(test_tfbs)

with open(SAVED_MODEL_PATH, 'r') as jf:
    json = jf.read()

model = model_from_json(json)
model.load_weights(SAVED_WEIGHTS_PATH)
print("Loaded the saved model")
 
predictions = model.predict(tfbs_test)
#print(predictions)
results = []
for p in predictions:
    if p[0] > 0.5:
        results.append([1])
    else:
        results.append([0])
#print(results)
df = pd.DataFrame(data = results, columns = {"prediction"})
df.to_csv(path_or_buf = RESULTS, columns={"prediction"}, header=True, index=True, index_label="id")

