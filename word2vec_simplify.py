import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import re
import os

# filename
# colname to fit on
# embedding dim
# word2vec file
# output file

df = pd.read_csv('datas.txt', header=0, sep=';')
corpus = df['Abstract'].tolist()
embed_size = 200


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


texts = []

for text in corpus:
    text_clean = clean_str(text)
    texts.append(text_clean)

tokenizer = Tokenizer(num_words=None, oov_token=True)
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index

embeddings_index = {}
f = open(os.path.join('', 'vectors_clean.txt'), encoding="utf-8")
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

print("Word2vec size : {}".format(len(embeddings_index)))

words = []
for word, i in word_index.items():
    # get the vector corresponding to word i
    words.append(word)

with open('BioASQ_light_vectors.txt', 'w', encoding='utf-8') as fileout:
    dic = {}
    for word in words:
        coeffs = embeddings_index.get(word)
        if coeffs is not None:
            str_coeff = ' '.join(coeffs.tolist())
            dic[word] = str_coeff
    fileout.write("{} {}\n".format(len(dic), embed_size))
    print("New word2vec file shape = {} ; {}\n".format(len(dic), embed_size))
    for word in dic:
        fileout.write("{} {}\n".format(word, dic[word]))

print("Done")


