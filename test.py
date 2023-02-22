from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.utils import pad_sequences
import numpy as np

embedding_matrix = np.load('./data/embedding_matrix.npy')

#build and load model
model = Sequential()
model.add(Embedding(12234,20,input_length=50,weights=[embedding_matrix],trainable=True))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(3,activation="softmax"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['acc'])

model.load_weights('./pretrained/model_001.h5')

#preprocessing
labels = ['positive', 'natural', 'negative']
message = [str(input('Input message: '))]
tokenizer = Tokenizer(num_words=500000, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
seq = tokenizer.texts_to_sequences(message)
padded = pad_sequences(seq, maxlen=50, dtype='int32', value=0)

pred = model.predict(padded)
print('--->', labels[np.argmax(pred)])
