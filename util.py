import os
import pickle
import numpy as np


def get_embedding(word_to_id, dataset):
    emb_file = '../glove.840B.300d.txt'
    pickle_file = './embeddings/%s_840B.pkl' % dataset
    dim = 300

    if not os.path.exists(pickle_file):  # First read word vector
        # embeddings = np.zeros((len(word_to_id) + 1, dim), dtype='float32')  # Initialize the word vector to 0
        embeddings = np.random.uniform(-0.2, 0.2, (len(word_to_id), dim))
        with open(emb_file, encoding='utf-8') as fp:
            for line in fp:
                elements = line.strip().split()
                word = elements[0]  # word
                if word in word_to_id:  # If the word is in the vocabulary, use the pre-training result
                    try:
                        embeddings[word_to_id[word]] = [float(v) for v in elements[1:]]
                    except ValueError:
                        pass
        # print("Find %s word embeddings!!" % n_emb)
        # pickle.dump(embeddings, open(pickle_file, 'wb'))  # Storage, easy to read next time
    else:
        embeddings = pickle.load(open(pickle_file, 'rb'))
    return embeddings
