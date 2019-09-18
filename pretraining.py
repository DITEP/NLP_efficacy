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
from sklearn.metrics import roc_curve


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


df = pd.read_csv('pretrain_data3.txt', header=0, sep=';', encoding='utf-8')
df = df.reset_index(drop=True)
df['Clinical_activity'] = np.nan
df.loc[df['Impact Factor'] <= 2.5, ['Clinical_activity']] = 0
df.loc[df['Impact Factor'] > 2.5, ['Clinical_activity']] = 1

to_pred = 'Clinical_activity'
labels = df[to_pred].values
df2 = df[['COMMON_DRUGBANK_ALIAS', 'Clinical_activity']].drop_duplicates()
corpus = df['Abstract'].tolist()

# parameters.txt
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
plt.savefig('number of words')

# cumulative distribution
plt.clf()
dx = bin_edges[1] - bin_edges[0]

cumulative = np.cumsum(hist)*dx
plt.plot(bin_edges[:-1], cumulative, c='blue')

plt.grid()
plt.title('Cumulative distribution \n of number of words \n by sentence')
plt.savefig('cumulative number of words')

# histogram number of sentences
plt.clf()
hist, bin_edges = np.histogram(sent_nums, bins=50)

plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], color='red', alpha=0.5)

plt.grid()
plt.title('Number of sentence \n by text')
plt.savefig('number of sentences')

# cumulative distribution
plt.clf()
dx = bin_edges[1] - bin_edges[0]

cumulative = np.cumsum(hist)*dx

plt.plot(bin_edges[:-1], cumulative, c='blue')

plt.grid()
plt.title('Cumulative distribution \n of number of sentences')
plt.savefig('cumulative number of sentences')

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
                    pass

word_index = tokenizer.word_index

args_pool = '0'

if args_pool == '0' or args_pool == '1':
    # create train and validation set
    np.random.seed(1234)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    x_train, x_val, x_test = np.split(data, [int(.8*len(df)), int(.9*len(df))])
    y_train, y_val, y_test = np.split(labels, [int(.8*len(df)), int(.9*len(df))])
    if args_pool == '1':
        # Split drugs names
        drugs = df['COMMON_DRUGBANK_ALIAS']  # drug name in data
        drugs = drugs[indices]
        drug_train, drug_val, drug_test = np.split(drugs, [int(.8*len(df)), int(.9*len(df))])
        drugs_list = list(set(drug_val))  # unique of the drug_val list0
        # unique of the dataframe
        df_pool = df[['COMMON_DRUGBANK_ALIAS', 'Clinical_activity']].drop_duplicates()
else:
    np.random.seed(1234)
    drugs = df['COMMON_DRUGBANK_ALIAS']
    list_of_drugs = df['COMMON_DRUGBANK_ALIAS'].drop_duplicates().reset_index(drop=True)
    indices = np.arange(list_of_drugs.shape[0])
    np.random.shuffle(indices)
    list_of_drugs = list_of_drugs[indices]
    train, val, test = np.split(list_of_drugs, [int(.7 * len(list_of_drugs)), int(.9 * len(list_of_drugs))])
    y_train = df.loc[df['COMMON_DRUGBANK_ALIAS'].isin(train), to_pred]
    y_val = df.loc[df['COMMON_DRUGBANK_ALIAS'].isin(val), to_pred]
    y_test = df.loc[df['COMMON_DRUGBANK_ALIAS'].isin(test), to_pred]
    x_train = data[y_train.index]
    x_val = data[y_val.index]
    x_test = data[y_test.index]
    drug_train = drugs[y_train.index]
    drug_val = drugs[y_val.index]
    drug_test = drugs[y_test.index]
    drugs_list = list(set(drug_val))  # unique of the drug_val list
    # unique of the dataframe
    df_pool = df[['COMMON_DRUGBANK_ALIAS', 'Clinical_activity']].drop_duplicates()


REG_PARAM = 1e-10
l2_reg = regularizers.l2(REG_PARAM)

embedding_matrix = np.zeros((len(word_index) + 1, embed_size))


embeddings_index = {}
f = open(os.path.join('', 'BioASQ_light_vectors_pretraining.txt'), encoding="utf-8")
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

# model
embedding_layer = Embedding(len(word_index) + 1,
                            embed_size,
                            weights=[embedding_matrix],
                            input_length=max_senten_len,
                            trainable=False)

# Words level attention model
word_input = Input(shape=(max_senten_len,), dtype='float32', name='word_input')
word_sequences = embedding_layer(word_input)
word_lstm = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2_reg, recurrent_dropout=0.5, dropout=0.52, name='word_lstm'))(word_sequences)
word_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg, name='word_dense'))(word_lstm)
word_att = Dropout(0.5)(AttentionWithContext(name='word_att')(word_dense))
wordEncoder = Model(word_input, word_att, name='wordEncoder')

# Sentence level attention model
sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32', name='sent_input')
sent_encoder = TimeDistributed(wordEncoder, name='sent_encoder')(sent_input)
sent_lstm = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2_reg, recurrent_dropout=0.5, dropout=0.52, name='sent_lstm'))(sent_encoder)
sent_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg, name='sent_dense'))(sent_lstm)
sent_att = Dropout(0.5)(AttentionWithContext(name='sen_att')(sent_dense))
preds = Dense(1, activation='sigmoid', name='preds')(sent_att)
model = Model(sent_input, preds, name='model')
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
print(model.summary())

# f1-score computation
metrics = Metrics()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=64, callbacks=[metrics])
model.save('pretrain_model.h5')
model.save_weights('pretrain_weights.h5')

# summarize history for accuracy
plt.clf()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy')

# summarize history for loss
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss')

# summarize history for F1-score
plt.clf()
plt.plot(metrics.val_f1s)
plt.title('model F1-score')
plt.ylabel('F1-score')
plt.xlabel('epoch')
plt.savefig('f1_score')


# output predict by the model
pred = model.predict(x_val)
flat_pred = [item for sublist in pred.tolist() for item in sublist]

if args_pool == '1' or args_pool == '2':
    y_true = []
    y_pred = []
    drug_val = drug_val.tolist()
    for drug in drugs_list:
        y_true.append(df_pool.loc[df_pool['COMMON_DRUGBANK_ALIAS'] == drug, to_pred].tolist())
        predictions = []
        for i in range(len(drug_val)):
            if drug_val[i] == drug:
                predictions.append(flat_pred[i])
        pool = sum(predictions) / len(predictions)
        y_pred.append([pool])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
else:
    y_pred = pred
    y_true = y_val
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred.ravel())


y_pred[y_pred > 0.5] = 1
y_pred[y_pred != 1] = 0


def plot_confusion_matrix(y_t, y_p,
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
    cm = confusion_matrix(y_t, y_p)
    # Only use the labels that appear in the data
    classes = unique_labels(y_t, y_p)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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


plt.clf()
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_true, y_pred,
                      title='Confusion matrix, without normalization')
plt.savefig('matrix')
