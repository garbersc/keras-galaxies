import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import load_data
from lsuv_init import LSUVinit

model = Sequential()
model.add(Dense(100, activation='sigmoid', input_dim=10))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


predictions = load_data.load_gz(
    'predictions/final/augmented/valid/try_10cat_wMaxout_next_next_next.npy.gz')

y_train = np.load("data/solutions_train_10cat.npy")
y_valid = y_train[-1 * len(predictions):]

target = [float(np.argmax(pred) == np.argmax(val))
          for pred, val in zip(predictions, y_valid)]

print model.summary()

# LSUVinit(model, [predictions[:100], target[:100]], batch_size=100)

print np.shape(predictions)
print predictions[:5]
print np.shape(target)
print target[:5]
print np.mean(target)

model.fit(predictions, target, epochs=1000, batch_size=30)

model.save_weights('analysis/final/fit_10cat_qual_weights.h5')
