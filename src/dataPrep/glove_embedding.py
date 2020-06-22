import wget
import os
import glob
import pathlib

PATH = pathlib.Path(__file__).parent.absolute()
import sys
import zipfile
import numpy as np

sys.path.append("./src/")

if not os.path.exists(os.path.join(PATH, "glove")):
    os.makedirs(os.path.join(PATH, "glove"))
    wget.download(
        "http://nlp.stanford.edu/data/glove.6B.zip", out=os.path.join(PATH, "glove")
    )
    with zipfile.ZipFile(
        os.path.join(os.path.join(PATH, "glove"), "glove.6B.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(os.path.join(PATH, "glove"))


def read_embedding_file(vocab, dim):
    embeddings_dict = {}
    with open(
        os.path.join(os.path.join(PATH, "glove"), f"glove.6B.{str(dim)}d.txt"), "r"
    ) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    matrix_len = len(vocab.word2idx)
    weights_matrix = np.zeros((matrix_len, dim))
    words_found = 0

    for i, word in enumerate(vocab.idx2word):
        try:
            weights_matrix[i] = embeddings_dict[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(dim,))
    return weights_matrix
