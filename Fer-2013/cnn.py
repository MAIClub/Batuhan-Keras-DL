import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

label_count = 7
batch_size = 128
epochs = 100
width, height = 48, 48


x = np.load('data.npy')
y = np.load('labels.npy')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=74)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=49)

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
#%%

model = Sequential()


model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), kernel_regularizer=l2(0.015)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))

model.add(Flatten())


model.add(Dense(64, activation='tanh'))
model.add(Dense(2 * 64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(label_count, activation='softmax'))


model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.003), metrics=['accuracy'])

model.summary()

filepath = "~/Documents/AI_Related/emotion_recog_fer/savedmodels/Model_1/modelcp.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

trained = model.fit(np.array(X_train), np.array(y_train),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(np.array(X_valid), np.array(y_valid)),
                    shuffle=True,
                    callbacks=callbacks_list)

print("****************************************************************************")
print("train completed")
