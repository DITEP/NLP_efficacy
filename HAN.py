import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
import re
import os
import matplotlib.pyplot as plt
from nltk import tokenize
import seaborn as sns


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

# https://github.com/Hsankesara/DeepResearch/blob/master/Hierarchical_Attention_Network/attention_with_context.py
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


df = pd.read_csv('datas.txt', header=0, sep=';')
labels = df['Clinical_activity'].values
corpus = df['Abstract'].tolist()

max_features = 200000  # maximum number of words to keep, based on word frequency
max_senten_len = 40  # to tune : ensure the security
max_senten_num = 6  # to tune : ensure the security
embed_size = 100  # to tune : ensure the security 
VALIDATION_SPLIT = 0.2


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


list_sentences = []
texts = []

sent_lens = []
sent_nums = []
# create a list of list (list_sentences) that contains for each text of the corpus a list of sentences
# [['sentence1', 'sentence2', ..., 'sentence n'], ['sentence1', ..., 'sentence n']]
for text in corpus:
    text_clean = clean_str(text)
    texts.append(text_clean)
    # tokenize the sentences
    sentences = tokenize.sent_tokenize(text_clean)
    list_sentences.append(sentences)
    # number of sentences by text
    sent_nums.append(len(sentences))
    for sent in sentences:
        # length of sentence
        sent_lens.append(len(text_to_word_sequence(sent)))


# set the max length of sentences and the max number of sentences in each text
# max_senten_len = max(sent_lens)
# max_senten_num = max(sent_nums)

sns.distplot(sent_lens, bins=200)
plt.show()
sns.distplot(sent_nums)
plt.show()

tokenizer = Tokenizer(num_words=max_features, oov_token=True)
tokenizer.fit_on_texts(texts)

# Create 3D tensor:
# the first dimension represents the documents
# the second one represents each sentence in a document
# the last one represents each word in a sentence.
data = np.zeros((len(texts), max_senten_num, max_senten_len), dtype='int32')
for i, sentences in enumerate(list_sentences):
    for j, sentence in enumerate(sentences):
        if j < max_senten_num:
            wordTokens = text_to_word_sequence(sentence)
            k = 0
            for _, word in enumerate(wordTokens):
                try:
                    if k < max_senten_len and tokenizer.word_index[word] < max_features:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k+1
                except:
                    print(word)
                    pass

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

print('Shape of data tensor:', data.shape)
print('Shape of labels tensor:', labels.shape)

# create train and validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
# print('Number of positive and negative reviews in training and validation set')
# print(y_train.sum(axis=0).tolist())
# print(y_val.sum(axis=0).tolist())
print('Test and validation set done')

REG_PARAM = 1e-13
l2_reg = regularizers.l2(REG_PARAM)
EMBEDDING_DIM = 100

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

embeddings_index = {}
f = open(os.path.join('', 'word2vec_abstract.txt'), encoding="utf-8")
for line in f:
    # extract each combination of word/coefficient
    values = line.split()
    # extract word
    word = values[0]
    # extract corresponding vector
    coefs = np.asarray(values[1:])
    # build a dictionary word : vector
    embeddings_index[word] = coefs
f.close()

for word, i in word_index.items():
    if i > num_words:
        continue
    # get the vector corresponding to word i
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros
        # replace by the corresponding word vector
        embedding_matrix[i] = embedding_vector

print("Number of words :", num_words)

embedding_layer = Embedding(len(word_index) + 1,
                            embed_size,
                            weights=[embedding_matrix],
                            input_length=max_senten_len,
                            trainable=False)

word_input = Input(shape=(max_senten_len,), dtype='float32')
word_sequences = embedding_layer(word_input)
word_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(word_sequences)
word_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(word_lstm)
word_att = AttentionWithContext()(word_dense)
wordEncoder = Model(word_input, word_att)

sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
sent_encoder = TimeDistributed(wordEncoder)(sent_input)
sent_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(sent_encoder)
sent_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(sent_lstm)
sent_att = Dropout(0.5)(AttentionWithContext()(sent_dense))
preds = Dense(1, activation='sigmoid')(sent_att)
model = Model(sent_input, preds)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.003), metrics=['acc'])
print(model.summary())

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15, batch_size=64)

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