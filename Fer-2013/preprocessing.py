import numpy as np
import pandas as pd 


dataset = pd.read_csv("fer2013.csv")

width, height = 48, 48 

data = dataset['pixels'].tolist()

X = []

for j in data:
    x = [int(i) for i in j.split(' ')]
    x = np.asarray(x).reshape(width,height)
    X.append(x.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

y = pd.get_dummies(dataset['emotion']).values

np.save('data',X)
np.save('labels', y)

print('completed...')


