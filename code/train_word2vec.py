from __future__ import division
import os
import time
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from scipy.spatial.distance import cosine


def cosine_sim(v1, v2):
    """
    Computes cosine similarity (1 - cosine_distance).

    :param v1: vector 1
    :param v2: vector 2
    :return: cosine similarity between v1 and v2
    """
    return 1 - cosine(v1, v2)


def load_model(model_path):
    """
    Returns loaded Word2vec model.

    :param model_path: path to trained model
    :return: loaded model
    """
    return Word2Vec.load(model_path)


def train_word2vec(data,
                   save_path,
                   model_name="my_model",
                   num_epoch=10,
                   vector_dim=300,
                   window_size=10,
                   min_frequency=1,
                   workers=1):
    """
    Trains a word2vec model.

    :param data: input texts (list of list of words)
    :param save_path: path to save trained model to
    :param model_name: name of your model (default 'my_model')
    :param num_epoch: number of epochs to train model for (default 10)
    :param vector_dim: dimension of word embeddings (default 300)
    :param window_size: context window size (default 10)
    :param min_frequency: minimum frequency to consider words in corpus (default 1)
    :param workers: number of workers to train your model (default 1)
    :return: trained word2vec KeyedVectors model
    """
    print("Train word2vec model '{}'".format(model_name))

    model = Word2Vec(data,
                     size=vector_dim,
                     window=window_size,
                     iter=num_epoch,
                     min_count=min_frequency,
                     workers=workers)

    print("Save word2vec model '{}' at {}".format(model_name, save_path))
    model.save("{}/{}.model".format(save_path, model_name))

    return model


if __name__ == "__main__":
    # one of train, load
    mode = 'train'
    w2v_model = None

    if mode == 'train':
        model_save_path = "./models/word2vec/w2v_model_{}".format(time.strftime("%Y%m%d-%H%M%S"))

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        w2v_model = train_word2vec(common_texts, model_save_path, vector_dim=300, window_size=5)
    elif mode == 'load':
        load_path = ""
        w2v_model = load_model(load_path)
    else:
        print("Wrong mode specified. Exit.")
        exit(0)

    # Example
    w_1 = 'trees'
    w_2 = 'graph'
    v_1 = w2v_model.wv[w_1]
    v_2 = w2v_model.wv[w_2]

    print("Cosine distance between '{}' and '{}': {}".format(w_1, w_2, cosine_sim(v_1, v_2)))
