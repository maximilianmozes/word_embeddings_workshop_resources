from __future__ import division
import os
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


def load_glove():
    """
    Returns a gensim KeyedVectors model with pre-trained Glove word embeddings.

    :return model: KeyedVectors model
    """
    print("Loading Glove model")

    path = "../data/glove.840B.300d.txt"
    g2w2v = "../data/g2w2v.txt"

    if not os.path.exists(g2w2v):
        print("Convert Glove to word2vec format")
        glove2word2vec(glove_input_file=path, word2vec_output_file=g2w2v)

    glove_model = KeyedVectors.load_word2vec_format(g2w2v, binary=False)

    return glove_model


def load_word2vec():
    """
    Returns a gensim KeyedVectors model with pre-trained word2vec word embeddings.

    :return model: KeyedVectors model
    """
    print("Loading word2vec model")

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
