from __future__ import division
import os
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
from gensim.models import KeyedVectors
from unidecode import unidecode


def load_word2vec(limit=-1):
    """
    Returns an embedding matrix of pre-trained word2vec word vectors together with two word-index association
    mappings.

    :param limit: number of embeddings to load into memory (default=-1 -> no limit)
    :return embeddings: embedding matrix of shape (N, D)
    :return tokens: list of words in indexed order
    """
    print("Loading word2vec Model")

    path = "../data/GoogleNews-vectors-negative300.bin"
    w2v_model = KeyedVectors.load_word2vec_format(path, binary=True)
    embeddings = []
    tokens = []

    for idx, (word, _) in enumerate(w2v_model.wv.vocab.items()):
        if idx == limit:
            break

        tokens.append(unidecode(word))
        embeddings.append(w2v_model.wv[word])

    return np.array(embeddings), tokens


def load_glove_raw(limit=-1):
    """
    Returns an embedding matrix of pre-trained GloVe word vectors together with two word-index association
    mappings.

    :param limit: number of embeddings to load into memory (default=-1 -> no limit)
    :return embeddings: embedding matrix of shape (N, D)
    :return tokens: list of words in indexed order
    """
    print("Loading Glove Model")

    path = "../data/glove.840B.300d.txt"
    data = open(path, 'r')
    embeddings = []
    tokens = []

    for idx, line in enumerate(data):
        if idx == limit:
            break

        row = line.strip().split(' ')
        word = row[0]
        embedding = np.array([float(val) for val in row[1:]])

        try:
            tokens.append(word.encode("utf-8"))
            embeddings.append(embedding)
        except UnicodeDecodeError:
            pass

    print("Loaded {} embeddings into memory".format(len(embeddings)))

    return np.array(embeddings), tokens


if __name__ == "__main__":
    # one of glove, word2vec
    model = 'glove'

    writer_dir = "./tensorboardX_visualize/{}/run_{}".format(model, time.strftime("%Y%m%d-%H%M%S"))
    word_limit = 10000

    embedding_matrix = None
    words = None

    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)

    writer = SummaryWriter(writer_dir)

    if model == 'glove':
        embedding_matrix, words = load_glove_raw(word_limit)
    elif model == 'word2vec':
        embedding_matrix, words = load_word2vec(word_limit)
    else:
        print("Wrong mode specified. Exit.")
        exit(0)

    writer.add_embedding(
        torch.FloatTensor(embedding_matrix),
        metadata=words,
        global_step=0
    )
