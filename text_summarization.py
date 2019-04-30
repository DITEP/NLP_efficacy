# let's begin
import pandas as pd
df = pd.read_csv('datas3_group.txt', header=0, sep=';', encoding='utf-8')
df = df.reset_index(drop=True)

LENGTH_SUMMARY = 80

summary_all = []
for drug in range(df.shape[0]):
    print("Iteration : {} / {}".format(drug, df.shape[0]))
    from nltk.tokenize import sent_tokenize
    s = df['Abstract'][drug]

    import re
    regex = re.compile("\s(?:[A-Za-z]\.){2,}")
    resultat = regex.findall(s)
    resultat = list(set(resultat))
    for r in resultat:
        r_clean = r.replace('.', '')
        s = s.replace(r, r_clean)

    sentences = sent_tokenize(s)

    from nltk.tokenize import word_tokenize
    print(len(sentences))
    sentences = [s for s in sentences if len(word_tokenize(s)) > 5]
    print(len(sentences))
    if len(sentences) < LENGTH_SUMMARY:
        summary = []
        for s in sentences:
            summary.append(s)
        print("Summary done")
        summary_all.append(summary)
        continue
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

    clean_sentences = [' '.join([w for w in word_tokenize(s) if not w in stop_words]) for s in clean_sentences]
    print("Stop words cleaning done")
    import numpy as np

    # Extract word vectors
    word_embeddings = {}
    f = open('BioASQ_light_vectors.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    # Tf-idf
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf_idf_vect = TfidfVectorizer(stop_words=None)
    final_tf_idf = tf_idf_vect.fit_transform(clean_sentences)
    tfidf_feat = tf_idf_vect.get_feature_names()

    sentence_matrix = np.empty((0, 200), int)
    for i, sent in enumerate(clean_sentences):
        if len(sent) != 0:
            sum_embed = np.zeros((200,))
            weight_sum = 0
            for w in sent.split():
                try:
                    tfidf = final_tf_idf[i, tfidf_feat.index(w)]
                    embed = word_embeddings.get(w, np.zeros((200,)))
                    embed = np.array(embed)
                    sum_embed = sum_embed + (embed*tfidf)
                    weight_sum += tfidf
                except:
                    pass
            v = sum_embed/(weight_sum+0.001)
        else:
            v = np.zeros((200,))
        sentence_matrix = np.vstack([sentence_matrix, v])

    print("Sentence embeddings done")
    print(sentence_matrix.shape)
    # similarity matrix
    sentence_matrix = sentence_matrix.transpose()
    d = sentence_matrix.T @ sentence_matrix
    norm = ((sentence_matrix * sentence_matrix).sum(0, keepdims=True) ** .5) + 0.001
    sim_mat = 1 - d / norm / norm.T

    print("Similarity matrix done")
    print(sim_mat.shape)
    import networkx as nx

    nx_graph = nx.from_numpy_array(sim_mat)
    print("Graph done")
    scores = nx.pagerank(nx_graph)
    print("Scores done")
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    # Extract top 10 sentences as the summary
    summary = []
    for i in range(LENGTH_SUMMARY):
        summary.append(ranked_sentences[i][1])
    print("Summary done")
    summary = ' '.join(summary)
    summary_all.append(summary)
df['summary'] = summary_all
df.to_csv('datas3_summary.txt', sep=';', encoding='utf-8')
print("Dataframe done")
