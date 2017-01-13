python lstm.py


Using Theano backend.
/home/xsede/users/xs-nksg/.local/lib/python2.7/site-packages/hyperas/optim.py
/home/xsede/users/xs-nksg/.local/lib/python2.7/site-packages/hyperas/optim.py
/home/xsede/users/xs-nksg/.local/lib/python2.7/site-packages/hyperas/optim.py
lstm.py
>>> Imports:
from __future__ import print_function

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    from keras.preprocessing import sequence
except:
    pass

try:
    from keras.datasets import imdb
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.layers.embeddings import Embedding
except:
    pass

try:
    from keras.layers.recurrent import LSTM
except:
    pass

try:
    from keras.callbacks import EarlyStopping, ModelCheckpoint
except:
    pass

>>> Hyperas search space:

def get_space():
    return {
        'Dropout': hp.uniform('Dropout', 0, 1),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
    }

>>> Data
  1:
  2: maxlen = 100
  3: max_features = 20000
  4:
  5: print('Loading data...')
  6: (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
  7: print(len(X_train), 'train sequences')
  8: print(len(X_test), 'test sequences')
  9:
 10: print("Pad sequences (samples x time)")
 11: X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
 12: X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
 13: print('X_train shape:', X_train.shape)
 14: print('X_test shape:', X_test.shape)
 15:
 16:
 17:
 18:
>>> Resulting replaced keras model:

  1: def keras_fmin_fnct(space):
  2:
  3:     model = Sequential()
  4:     model.add(Embedding(max_features, 128, input_length=maxlen))
  5:     model.add(LSTM(128))
  6:     model.add(Dropout(space['Dropout']))
  7:     model.add(Dense(1))
  8:     model.add(Activation('sigmoid'))
  9:
 10:     model.compile(loss='binary_crossentropy',
 11:                   optimizer='adam',
 12:                   metrics=['accuracy'])
 13:
 14:     early_stopping = EarlyStopping(monitor='val_loss', patience=4)
 15:     checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
 16:                                    verbose=1,
 17:                                    save_best_only=True)
 18:
 19:     model.fit(X_train, y_train,
 20:               batch_size=space['batch_size'],
 21:               nb_epoch=1,
 22:               validation_split=0.08,
 23:               callbacks=[early_stopping, checkpointer])
 24:
 25:     score, acc = model.evaluate(X_test, y_test, verbose=0)
 26:
 27:     print('Test accuracy:', acc)
 28:     return {'loss': -acc, 'status': STATUS_OK, 'model': model}
 29:
Loading data...
25000 train sequences
25000 test sequences
Pad sequences (samples x time)
X_train shape: (25000, 100)
X_test shape: (25000, 100)
Train on 23000 samples, validate on 2000 samples
Epoch 1/1
22912/23000 [============================>.] - ETA: 1s - loss: 0.4385 - acc: 0.7892Epoch 00000: val_loss improved from inf to 0.34510, saving mo23000/23000 [==============================] - 377s - loss: 0.4382 - acc: 0.7894 - val_loss: 0.3451 - val_acc: 0.8420
Test accuracy: 0.84096
Train on 23000 samples, validate on 2000 samples
Epoch 1/1
22976/23000 [============================>.] - ETA: 0s - loss: 0.4822 - acc: 0.7567Epoch 00000: val_loss improved from inf to 0.36107, saving mo23000/23000 [==============================] - 258s - loss: 0.4822 - acc: 0.7567 - val_loss: 0.3611 - val_acc: 0.8375
Test accuracy: 0.84328
Train on 23000 samples, validate on 2000 samples
Epoch 1/1
22976/23000 [============================>.] - ETA: 0s - loss: 0.4609 - acc: 0.7766Epoch 00000: val_loss improved from inf to 0.38023, saving mo23000/23000 [==============================] - 338s - loss: 0.4607 - acc: 0.7767 - val_loss: 0.3802 - val_acc: 0.8380
Test accuracy: 0.83676
Train on 23000 samples, validate on 2000 samples
Epoch 1/1
22976/23000 [============================>.] - ETA: 0s - loss: 0.4373 - acc: 0.7947Epoch 00000: val_loss improved from inf to 0.34915, saving mo23000/23000 [==============================] - 329s - loss: 0.4374 - acc: 0.7947 - val_loss: 0.3491 - val_acc: 0.8485
Test accuracy: 0.85204
Train on 23000 samples, validate on 2000 samples
Epoch 1/1
22976/23000 [============================>.] - ETA: 0s - loss: 0.4300 - acc: 0.8002Epoch 00000: val_loss improved from inf to 0.39696, saving mo23000/23000 [==============================] - 258s - loss: 0.4298 - acc: 0.8003 - val_loss: 0.3970 - val_acc: 0.8185
Test accuracy: 0.82324
Train on 23000 samples, validate on 2000 samples
Epoch 1/1
22912/23000 [============================>.] - ETA: 6s - loss: 0.4411 - acc: 0.7952 Epoch 00000: val_loss improved from inf to 0.39775, saving m23000/23000 [==============================] - 1821s - loss: 0.4411 - acc: 0.7953 - val_loss: 0.3978 - val_acc: 0.8225
Test accuracy: 0.8222
Train on 23000 samples, validate on 2000 samples
Epoch 1/1
22976/23000 [============================>.] - ETA: 0s - loss: 0.4325 - acc: 0.7943Epoch 00000: val_loss improved from inf to 0.36509, saving mo23000/23000 [==============================] - 247s - loss: 0.4322 - acc: 0.7944 - val_loss: 0.3651 - val_acc: 0.8345
Test accuracy: 0.8336
Train on 23000 samples, validate on 2000 samples
Epoch 1/1
22912/23000 [============================>.] - ETA: 6s - loss: 0.4748 - acc: 0.7714 Epoch 00000: val_loss improved from inf to 0.34959, saving m23000/23000 [==============================] - 1659s - loss: 0.4743 - acc: 0.7717 - val_loss: 0.3496 - val_acc: 0.8495
Test accuracy: 0.84144
Train on 23000 samples, validate on 2000 samples
Epoch 1/1
22976/23000 [============================>.] - ETA: 0s - loss: 0.4481 - acc: 0.7834Epoch 00000: val_loss improved from inf to 0.42220, saving mo23000/23000 [==============================] - 254s - loss: 0.4479 - acc: 0.7835 - val_loss: 0.4222 - val_acc: 0.7975
Test accuracy: 0.79984
Train on 23000 samples, validate on 2000 samples
Epoch 1/1
22976/23000 [============================>.] - ETA: 0s - loss: 0.4234 - acc: 0.8039Epoch 00000: val_loss improved from inf to 0.36021, saving mo23000/23000 [==============================] - 374s - loss: 0.4233 - acc: 0.8040 - val_loss: 0.3602 - val_acc: 0.8470
Test accuracy: 0.84752
{'Dropout': 0.4844455237320119, 'batch_size': 1}
