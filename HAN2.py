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
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels


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


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" - val_f1: {:.4f} - val_precision: {:.4f} - val_recall {:.4f}".format(_val_f1, _val_precision, _val_recall))
        return


df = pd.read_csv('datas.txt', header=0, sep=';')
labels = df['Clinical_activity'].values
corpus = df['Abstract'].tolist()

# parameters
max_features = 200000  # maximum number of words to keep, based on word frequency
max_senten_len = 50  # to tune : ensure the security
max_senten_num = 15  # to tune : ensure the security
embed_size = 200  # to tune : ensure the security
VALIDATION_SPLIT = 0.1


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

# histogram of sentence length (number of words by sentence)
hist, bin_edges = np.histogram(sent_lens, bins=50)

plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], color='red', alpha=0.5)

plt.grid()
plt.title('Number of words \n by sentence')
plt.show()

# cumulative distribution
dx = bin_edges[1] - bin_edges[0]

cumulative = np.cumsum(hist)*dx

plt.plot(bin_edges[:-1], cumulative, c='blue')

plt.grid()
plt.title('Cumulative distribution \n of number of words \n by sentence')
plt.show()

# histogram number of sentences

hist, bin_edges = np.histogram(sent_nums, bins=50)

plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], color='red', alpha=0.5)

plt.grid()
plt.title('Number of sentence \n by text')
plt.show()

# cumulative distribution
dx = bin_edges[1] - bin_edges[0]

cumulative = np.cumsum(hist)*dx

plt.plot(bin_edges[:-1], cumulative, c='blue')

plt.grid()
plt.title('Cumulative distribution \n of number of sentences')
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
np.random.seed(1234)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Split train 80%, val 10%, test 10%
x_train, x_val, x_test = np.split(data, [int(.8*len(df)), int(.9*len(df))])
y_train, y_val, y_test = np.split(labels, [int(.8*len(df)), int(.9*len(df))])

print('Test and validation set done')

REG_PARAM = 1e-13
l2_reg = regularizers.l2(REG_PARAM)

embedding_matrix = np.zeros((len(word_index) + 1, embed_size))


embeddings_index = {}
f = open(os.path.join('', 'BioASQ_light_vectors.txt'), encoding="utf-8")
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

absent_words = 0
for word, i in word_index.items():
    # get the vector corresponding to word i
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        # replace by the corresponding word vector
        embedding_matrix[i] = embedding_vector
    else:
        absent_words += 1

print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)), '% of total words')
print("Embedding tensor shape :", embedding_matrix.shape)

# model
embedding_layer = Embedding(len(word_index) + 1,
                            embed_size,
                            weights=[embedding_matrix],
                            input_length=max_senten_len,
                            trainable=False)

# Words level attention model
word_input = Input(shape=(max_senten_len,), dtype='float32')
word_sequences = embedding_layer(word_input)
word_lstm = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2_reg, recurrent_dropout=0.6, dropout=0.5))(word_sequences)
word_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(word_lstm)
word_att = AttentionWithContext()(word_dense)
wordEncoder = Model(word_input, word_att)

# Sentence level attention model
sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
sent_encoder = TimeDistributed(wordEncoder)(sent_input)
sent_lstm = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2_reg, recurrent_dropout=0.6))(sent_encoder)
sent_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(sent_lstm)
sent_att = Dropout(0.6)(AttentionWithContext()(sent_dense))
preds = Dense(1, activation='sigmoid')(sent_att)
model = Model(sent_input, preds)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0004), metrics=['acc'])
print(model.summary())

# f1-score computation
metrics = Metrics()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=64, callbacks=[metrics])

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

# summarize history for F1-score
plt.plot(metrics.val_f1s)
plt.title('model F1-score')
plt.ylabel('F1-score')
plt.xlabel('epoch')
plt.show()

# ROC curve and AUC display
from sklearn.metrics import roc_curve
y_pred = model.predict(x_val)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val, y_pred.ravel())
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='HAN (AUC = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

y_pred[y_pred > 0.5] = 1
y_pred[y_pred != 1] = 0


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_val, y_pred,
                      title='Confusion matrix, without normalization')
plt.show()

# Plot normalized confusion matrix
plot_confusion_matrix(y_val, y_pred, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
