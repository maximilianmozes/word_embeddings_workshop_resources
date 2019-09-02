from __future__ import division
import os
import time
from scipy.spatial.distance import cosine
from gensim.test.utils import common_texts
from glove import Glove, Corpus


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
    return Glove.load(model_path)


def train_glove(data,
                save_path,
                model_name='my_model',
                num_epoch=5,
                vector_dim=300,
                window_size=10,
                workers=1,
                alpha=0.75,
                max_count=100,
                learning_rate=0.01,
                ):
    """
    Trains a GloVe model.

    :param data: input texts (list of list of words)
    :param save_path: path to save trained model to
    :param model_name: name of your model (default 'my_model')
    :param num_epoch: number of epochs to train model for (default 10)
    :param vector_dim: dimension of word embeddings (default 300)
    :param window_size: context window size (default 10)
    :param workers: number of workers to train your model (default 1)
    :param alpha: alpha value for weighting function (default 0.75)
    :param max_count: x_max value for weighting function (default 100)
    :param learning_rate: leaning rate to train the model (default 0.01)
    :return: trained GloVe model
    """
    print("Train GloVe model '{}'".format(model_name))

    my_corpus = Corpus()
    my_corpus.fit(data, window=window_size)

    glove = Glove(no_components=vector_dim, learning_rate=learning_rate, alpha=alpha, max_count=max_count)
    glove.fit(my_corpus.matrix, epochs=num_epoch, no_threads=workers, verbose=False)
    glove.add_dictionary(my_corpus.dictionary)

    print("Save GloVe model '{}' at {}".format(model_name, save_path))
    glove.save("{}/{}.model".format(save_path, model_name))

    return glove


if __name__ == "__main__":
    # one of train, load
    mode = 'train'
    glove_model = None

    if mode == 'train':
        model_save_path = "./models/glove/glove_model_{}".format(time.strftime("%Y%m%d-%H%M%S"))

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        glove_model = train_glove(common_texts, model_save_path, vector_dim=300, window_size=5)
    elif mode == 'load':
        load_path = ""
        glove_model = load_model(load_path)
    else:
        print("Wrong mode specified. Exit.")
        exit(0)

    # Example
    w_1 = 'trees'
    w_2 = 'graph'
    v_1 = glove_model.word_vectors[glove_model.dictionary[w_1]]
    v_2 = glove_model.word_vectors[glove_model.dictionary[w_2]]

    print("Cosine distance between '{}' and '{}': {}".format(w_1, w_2, cosine_sim(v_1, v_2)))
