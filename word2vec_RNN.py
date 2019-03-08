# -*- coding: utf-8 -*-
"""phase1_prediction
"""

import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.optimizers import Adam
import nltk

nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv('datas.txt', header=0, sep=';')
data.head()

X = data.loc[:, 'Abstract']
y = data.loc[:, 'Clinical_activity']
total_reviews = X.tolist()
max_length = max([len(s.split()) for s in total_reviews])

EMBEDDING_DIM = 100

review_lines = list()
lines = data['Abstract'].values.tolist()

for line in lines:
    tokens = word_tokenize(line)
    # convert to lower
    tokens = [w.lower() for w in tokens]
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    review_lines.append(words)

embeddings_index = {}
f = open(os.path.join('', 'word2vec_abstract.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()

# Vectorize the text samples into a 2D integer tensor
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(review_lines)
sequences = tokenizer_obj.texts_to_sequences(review_lines)

# pad sequences
word_index = tokenizer_obj.word_index
print("Found %s unique tokens." % len(word_index))

review_pad = pad_sequences(sequences, maxlen=max_length)
sentiment = data['Clinical_activity'].values
print('Shape of abstract tensor:', review_pad.shape)
print('Shape of clinical tensor:', sentiment.shape)

EMBEDDING_DIM = 100

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros
        embedding_matrix[i] = embedding_vector

print("Number of words :", num_words)

# define model
model = Sequential()
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length,
                            trainable=False)
model.add(embedding_layer)
model.add(Bidirectional(GRU(units=32, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])

# Changer les paramètres de validation et de test : 0.75 ; 0.125 ; 0.125
VALIDATION_SPLIT = 0.2

indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
sentiment = sentiment[indices]
num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])

X_train_pad = review_pad[:-num_validation_samples]
y_train = sentiment[:-num_validation_samples]
X_test_pad = review_pad[-num_validation_samples:]
y_test = sentiment[-num_validation_samples:]

history = model.fit(X_train_pad, y_train, batch_size=64, epochs=15, validation_data=(X_test_pad, y_test), verbose=1)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()