from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from sklearn.model_selection import train_test_split
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
#nltk.download('stopwords')
#nltk.download('punkt')

df = pd.read_csv('./data/pre_data.csv', delimiter=',', encoding='latin-1')
train, test = train_test_split(df, test_size=0.000001 , random_state=42)

#list of stopwords
sws = set(stopwords.words('english'))

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            #if len(word) < 0:
            if len(word) <= 0 or word in sws:
                continue
            tokens.append(word.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)

# The maximum number of words to be used. (most frequent)
max_fatures = 500000

# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 50

#tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Message'].values)
X = tokenizer.texts_to_sequences(df['Message'].values)
X = pad_sequences(X, MAX_SEQUENCE_LENGTH)
#print('Found %s unique tokens.' % len(X))


'''
#model for calculating word vector similarity

d2v_model = Doc2Vec(dm=1, dm_mean=1, vector_size=20, window=5, min_count=1, workers=8,ns_exponent=5, alpha=0.065, min_alpha=0.065)
d2v_model.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    d2v_model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    d2v_model.alpha -= 0.002
    d2v_model.min_alpha = d2v_model.alpha


d2v_model.save('./data/d2v_model.kv')
'''

d2v_model = Doc2Vec.load('./data/d2v_model.kv')

# save the vectors in a new matrix
embedding_matrix = np.zeros((len(d2v_model.wv)+ 1, 20))

for i, vec in zip(d2v_model.dv.index_to_key, d2v_model.dv):
    while i in vec <= 1000:
        embedding_matrix[i]=vec

np.save('./data/embedding_matrix.npy', embedding_matrix)

model = Sequential()
model.add(Embedding(len(d2v_model.wv)+1,20,input_length=X.shape[1],weights=[embedding_matrix],trainable=True))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(3,activation="softmax"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['acc'])

#print(model.summary())
print(len(d2v_model.wv)+1)
print(X.shape[1])


Y = pd.get_dummies(df['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)
history=model.fit(X_train, Y_train, epochs =50, batch_size=32, verbose = 2)


#plot
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('./pretrained/model_accuracy.png')

# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('./pretrained/model_loss.png')

#evaluate model
train_acc = model.evaluate(X_train, Y_train)
test_acc = model. evaluate(X_test, Y_test)

model.save('./pretrained/model_001.h5')