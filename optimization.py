from hyperopt import STATUS_OK

from keras.datasets import imdb 
vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)

word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}

from keras.preprocessing import sequence

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout,GRU

#Define the model
def objective(params):
    model=Sequential()
    model.add(Embedding(vocabulary_size, int(params['embedding_size']), input_length=max_words))
    model.add(GRU(int(params['units']), dropout = params['dropout']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    X_valid, y_valid = X_train[:int(params['batch_size'])], y_train[:int(params['batch_size'])]
    X_train2, y_train2 = X_train[int(params['batch_size']):], y_train[int(params['batch_size']):]
    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=int(params['batch_size']), epochs=int(params['num_epochs']))
    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracy = scores[1]
    loss = scores[0]
    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([accuracy, loss, params])
    of_connection.close()
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

from hyperopt import hp
# Define the search space
space = {
    'embedding_size': hp.quniform('embedding_size', 30, 100, 1),
    'units': hp.quniform('units', 50, 150, 1),
    'dropout': hp.uniform('dropout', 0.1, 1.0),
    'num_epochs':hp.quniform('num_epochs', 1, 4, 1),
    'batch_size': hp.choice('batch_size', [64, 128, 512])
}

import csv

from hyperopt import tpe

# Algorithm
tpe_algorithm = tpe.suggest

from hyperopt import Trials

# Trials object to track progress
bayes_trials = Trials()

# File to save first results
out_file = 'gru_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['accuracy','loss', 'params'])
of_connection.close()

from hyperopt import fmin

MAX_EVALS = 200

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)
