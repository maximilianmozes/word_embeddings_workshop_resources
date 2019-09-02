from __future__ import division
import os
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def get_nearest_neighbors(word, embeddings, k=100):
    """
    Returns the k nearest neighbors of the input word (in terms of cosine distance).

    :param word: word of consideration (string)
    :param embeddings: KeyedVectors embeddings model
    :param k: number of nearest neighbors to return (default 100)
    :return: list of strings containing the word's k nearest neighbors
    """
    print("======== Retrieve {} nearest neighbors of {} ========".format(k, word))

    neighbors = [neighbor for (neighbor, score) in embeddings.most_similar(word, topn=k)]

    return neighbors


def get_nearest_neighbors_raw(word, embeddings, wtoi, itow, k=100):
    """
    Returns the k nearest neighbors of the input word (in terms of cosine distance).

    :param word: word of consideration (string)
    :param embeddings: embedding matrix of shape (N, D)
    :param wtoi: dictionary mapping words to indices
    :param itow: dictionary mapping indices to words
    :param k: number of nearest neighbors to return (default 100)
    :return: list of strings containing the word's k nearest neighbors
    """
    print("======== Retrieve {} nearest neighbors of {} ========".format(k, word))

    try:
        idx = wtoi[word]
        dot_p = embeddings.dot(embeddings[idx])
        idx_norm = np.linalg.norm(embeddings[idx])
        m_norm = np.linalg.norm(embeddings, axis=1)
        norms_p = np.multiply(idx_norm, m_norm)
        cos_dist = 1 - (dot_p * (1 / norms_p))
        sorted_idx = np.delete(np.argsort(cos_dist), [idx])
        sorted_idx = sorted_idx[:k].tolist()
        verb_neighbors = [itow[x] for x in sorted_idx]

        return verb_neighbors
    except KeyError:
        print("Word {} not in dictionary. Exit.".format(word))
        exit(0)


def load_glove_raw():
    """
    Returns an embedding matrix of pre-trained GloVe word vectors together with two word-index association
    mappings.

    :return embeddings: embedding matrix of shape (N, D)
    :return w2idx: dictionary mapping words to indices
    :return idx2w: dictionary mapping indices to words
    """
    print("Loading Glove Model")

    path = "../data/glove.840B.300d.txt"
    data = open(path, 'r')
    embeddings = []
    w2idx = {}
    idx2w = {}

    for idx, line in enumerate(data):
        row = line.strip().split(' ')
        word = row[0]
        embedding = np.array([float(val) for val in row[1:]])
        embeddings.append(embedding)
        w2idx[word] = idx
        idx2w[idx] = word

    print("Loaded {} embeddings into memory".format(len(embeddings)))

    return np.array(embeddings), w2idx, idx2w


def load_glove():
    """
    Returns a gensim KeyedVectors model with pre-trained GloVe word embeddings.

    :return model: KeyedVectors model
    """
    print("Loading GloVe Model")

    path = "../data/glove.840B.300d.txt"
    g2w2v = "../data/g2w2v.txt"

    if not os.path.exists(g2w2v):
        print("Convert GloVe to word2vec format")
        glove2word2vec(glove_input_file=path, word2vec_output_file=g2w2v)

    glove_model = KeyedVectors.load_word2vec_format(g2w2v, binary=False)

    return glove_model


def load_word2vec():
    """
    Returns a gensim KeyedVectors model with pre-trained word2vec word embeddings.

    :return model: KeyedVectors model
    """
    print("Loading word2vec Model")

    path = "../data/GoogleNews-vectors-negative300.bin"
    w2v_model = KeyedVectors.load_word2vec_format(path, binary=True)

    return w2v_model


if __name__ == "__main__":
    # one of word2vec, glove
    model = 'glove'
    embedding_model = None

    if model == 'word2vec':
        embedding_model = load_word2vec()
    elif model == 'glove':
        embedding_model = load_glove()
    else:
        print("Wrong model specified. Exit.")
        exit(0)

    examples = ['russia', 'france', 'paris', 'moscow', 'mouse', 'dog', 'cat', 'computer']

    for example in examples:
        print(get_nearest_neighbors(example, embedding_model, k=10))
