# Resources for the EURO CSS 2019 Workshop on Word Embeddings for Computational Social Science

This repository contains all data and code used for our EURO CSS 2019 workshop ["A Gentle Introduction to Word Embeddings for the Computational Social Sciences"](https://maximilianmozes.github.io/word-embeddings-workshop/).

All slides used during the workshop will be made available in the `slides` directory.

## Getting started

### Requirements
This repository is based on the [gensim word2vec](https://radimrehurek.com/gensim/models/word2vec.html) and [glove-python](https://github.com/maciejkula/glove-python) libraries. For detailed tutorials on how to use word embeddings in Python, see [here](https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92) (word2vec with gensim) and [here](https://medium.com/analytics-vidhya/word-vectorization-using-glove-76919685ee0b) (Glove with glove-python). 

The code runs on `Python 2.7.12`. You will need to use `pip` to install all required packages. We highly recommend to use a `virtualenv` for your setup. See [https://pypi.org/project/virtualenv/](https://pypi.org/project/virtualenv/) for how to install `virtualenv` on your machine or simply run `pip install virtualenv`. You can then create a new virtual environment with

```
$ virtualenv euro_css_word_embeddings
```

This creates a new virtual environment in your current directory (**note**: you can of course change the name of your virtual environment - `euro_css_word_embeddings` is just a suggestion). 

Afterwards, activate your virtual environment with

```
$ source euro_css_word_embeddings/bin/activate
```

Once you have set up your virtual environment, you can install all required dependencies by executing the following command:

```
$ pip install -r requirements.txt
```

**Remark**: Once you are finished, you can exit your `virtualenv` by simply running

```
$ deactivate
```

### Data: pre-trained embeddings

Download the pre-trained Glove and word2vec word embeddings with the following commands:

```
$ cd data
$ chmod +x download.sh
$ ./download.sh
```

## Visualising a pre-trained model

You can visualise pre-trained word embeddings using TensorBoard via the `tensorboardX` package. The code is available at `code/visualize.py`. You can visualise pre-trained embeddings with the following steps:

1. In `code/visualize.py`, select the model that you would like to visualise by specifying the `model` variable. Also specify the number of words that you would like to visualise using the `limit` variable.

2. Run the code using

   ```
   $ cd code
   $ python -u visualize.py
   ```

3. Navigate to the saved embedding object. The object should be present in `code/tensorboardX_visualize/[MODEL]/run_[DATE_TIME]` where `[MODEL]` is the model that you selected (step 1) and `[DATE_TIME]` is the date and time at which you executed the `visualize.py`script. Run

   ```
   $ cd code/tensorboardX_visualize/[MODEL]/run_[DATE_TIME]
   $ tensorboard --port=8008 --logdir .
   ```

   to launch TensorBoard.

4. Visit http://localhost:8008 to open TensorBoard in your browser. The embeddings should now be displayed.

## Running pre-trained examples

We furthermore provide code to execute a few examples using the pre-trained word2vec and Glove embeddings. Run

```
$ cd code
$ python -u examples.py
```

to load the pre-trained embeddings and compute cosine distances between specific word embeddings (see `code/examples.py` for more details).

## Train your own vector space models

Finally, you can train your own vector space models in Python using the code provided in this repository.

### Train a Glove model

To train your own Glove model, we use the `glove_python` software package. The code to train your model and load a pre-trained model can be found in `code/train_glove.py`. Here you can specify whether you would like to train a new model (set `mode="train"`) or load an existing model (`mode="load"`). Note that for the latter, you have to specify the path to your pre-trained model via the `load_path` variable. You can train a model using

```
$ cd code
$ python -u train_glove.py
```

The trained model will be saved under `models/glove/glove_model_[DATE_TIME]` and can then be loaded from there.

### Train a word2vec model

You can train your own word2vec model in a similar way as the Glove. To train a word2vec model, we make use of the `gensim`software package. The code for training a word2vec model can be found in `code/train_word2ve.py`. After specifying your parameters, run 

```
$ cd code
$ python -u train_word2vec.py
```

The trained model will be saved under `models/word2vec/w2v_model_[DATE_TIME]` and can then be loaded from there.

### Use your own data

Currently, we use a small set of texts to train the above models (for educational purposes). If you want to train the embedding models on your own data, you need to load your data into a Python object. This object should be a list of lists, where each list corresponds to a single document. Each element in this sublist then corresponds to a word in this document. Example (`common_texts` object of the `gensim.test.utils` module):

```python
my_data = [['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time'], ['eps', 'user', 'interface', 'system'], ['system', 'human', 'system', 'eps'], ['user', 'response', 'time'], ['trees'], ['graph', 'trees'], ['graph', 'minors', 'trees'], ['graph', 'minors', 'survey']]
```

Here, `my_data` would be a list of documents where each sublist represents the sequence of words comprising the document. 

Once you have created this list, you can train the embedding models by changing the first argument of the `train_word2vec` (in `code/train_word2vec.py`) or the `train_glove` (in `code/train_glove.py`) functions, i.e.

```python
w2v_model = train_word2vec(my_data, model_save_path, vector_dim=300, window_size=5)
```

or 

````python
glove_model = train_glove(my_data, model_save_path, vector_dim=300, window_size=5)
````

and then run the script as explained above.



 

 