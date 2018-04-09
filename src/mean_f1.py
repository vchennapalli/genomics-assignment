"""
Author - Vineeth Chennapoalli
Big Data Genomics Assignment
"""
import pandas as pd
DATA_FOLDER = '../data'
TRAIN_PATH = DATA_FOLDER + '/train_1.csv'
RESULTS = '../results/predictions.csv'

colnames = ['sequence', 'label']
data = pd.read_csv(TRAIN_PATH, names = colnames)
labels = data.label.tolist()
labels = labels[1921:]

colnames = ['id', 'prediction']
results = pd.read_csv(RESULTS, names = colnames)
predictions = results.prediction.tolist()
predictions = predictions[1:]
print(labels)
print(predictions)

tp, tn, fp, fn = 0, 0, 0, 0
print(len(labels))
print(len(predictions))
for i in range(480):
    if labels[i] == predictions[i]:
        if predictions[i] == 1:
            tp += 1
        else:
            tn += 1
    else:
        if predictions[i] == 0:
            fn += 1
        else:
            fp += 1


precision = float(tp) / (tp + fp)
recall    = float(tp) / (tp + fn)

f1_score = 2 * p * r / (p + r)
