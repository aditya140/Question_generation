from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import re
import wget
import os
import time
import glob
import pickle
import json
import pandas as pd
import shutil


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def max_length(tensor):
    return max(len(t) for t in tensor)


def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len:
        padded[:] = x[:max_len]
    else:
        padded[: len(x)] = x
    return padded


def get_torch_device(gpu=True):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


# conver the data to tensors and pass to the Dataloader
# to create an batch iterator


class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        # TODO: convert this into torch code is possible
        self.length = [np.sum(1 - np.equal(x, 0)) for x in X]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x, y, x_len

    def __len__(self):
        return len(self.data)


def load_glove_embedding(vocab, glove):

    nlp = glove
    vocab_size = len(vocab.word2idx)
    word_vec_size = 300
    embedding = np.zeros((vocab_size, word_vec_size))
    unk_count = 0

    print("=" * 100)
    print("Loading spacy glove embedding:")
    print("- Vocabulary size: {}".format(vocab_size))
    print("- Word vector size: {}".format(word_vec_size))

    for token, index in tqdm(vocab.word2idx.items()):
        if token == vocab.special["pad_token"]:
            continue
        elif token in [
            vocab.special["eos_token"],
            vocab.special["init_token"],
            vocab.special["unk_token"],
        ]:
            vector = np.random.rand(word_vec_size,)
        elif token in nlp.keys():
            vector = nlp[token]
        else:
            vector = embedding[vocab.word2idx[vocab.special["unk_token"]]]
            unk_count += 1

        embedding[index] = vector

    print("\n- Unknown word count: {}".format(unk_count))
    print("=" * 100 + "\n")

    return torch.from_numpy(embedding).float()


def download_glove(link="http://nlp.stanford.edu/data/glove.6B.zip"):
    print("Downloading glove")
    if not os.path.exists("./glove"):
        if not os.path.exists("glove.zip"):
            wget.download(link, "glove.zip")
        print("Extracting Glove")
        with zipfile.ZipFile("./glove.zip", "r") as zip_ref:
            zip_ref.extractall("glove")


import numpy as np
from tqdm import tqdm


def loadGloveModel(gloveFile="./glove/glove.6B.300d.txt"):
    if not os.path.exists(gloveFile):
        download_glove()
    print("Loading Glove Model")
    with open(gloveFile, "r") as f:
        model = {}
        for line in tqdm(f, position=0, leave=False):
            splitLine = line.split()
            try:
                word = " ".join(splitLine[0:-300])
                embedding = np.array([float(val) for val in splitLine[-300:]])
                model[word] = embedding
            except:
                print(f"Error in {splitLine[0]}")
                print(f"Vect : \n{splitLine[1:]}")
        print("Done.", len(model), " words loaded!")
    return model


def get_max_version(name):
    path = f"./src/saved_models/{name}/"
    versions = [int(i.split("/")[-2]) for i in glob.glob(path + "/*/")]
    return versions


def save_model(path, name, params, model, **kwargs):
    if "version" not in kwargs.keys():
        version = 0
        versions = [int(i.split("/")[-2]) for i in glob.glob(path + "/*/")]
        if versions != []:
            version = max(versions) + 1
    else:
        version = kwargs["version"]
    path = path + f"/{str(version)}/"
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), path + name)
    if "inpLang" in kwargs.keys():
        with open(path + "inpLang.p", "wb") as f:
            pickle.dump(kwargs["inpLang"], f)
    if "optLang" in kwargs.keys():
        with open(path + "optLang.p", "wb") as f:
            pickle.dump(kwargs["optLang"], f)
    with open(path + "hp.json", "w") as f:
        json.dump(params, f)
    if "test_df" in kwargs.keys():
        kwargs["test_df"].to_csv(path + "test_df.csv")


def load_model(name, version):
    path = f"./src/saved_models/{name}/{version}"
    with open(path + "/hp.json", "r") as f:
        hp = json.load(f)
    with open(path + "/inpLang.p", "rb") as f:
        inpLang = pickle.load(f)
    with open(path + "/optLang.p", "rb") as f:
        optLang = pickle.load(f)
    state_dict = torch.load(path + f"/{name}.pt")
    return state_dict, inpLang, optLang, hp


def load_test_df(name, version):
    path = f"./src/saved_models/{name}/{version}"
    df = pd.read_csv(path + "/test_df.csv", index_col=[0])
    return df


def save_test_df(df, name, version):
    path = f"./src/saved_models/{name}/{version}"
    df.to_csv(path + "/test_df.csv")
    print(" DataFrame Saved")


def save_metrics(metrics, name, version):
    path = f"./src/saved_models/{name}/{version}"
    with open(path + "/test_df_metrics.csv", "w") as fp:
        json.dump(metrics, fp)
    print("Metrics Saved")


def save_to_artifactory(name, version):
    pass


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def save_to_artifact(name, version):
    path = f"./src/saved_models/{name}/{version}"
    dest = f"/artifacts/{name}/{version}"
    if not os.path.exists(dest):
        os.makedirs(dest)
    copytree(path, dest)


def arg_copy(inp, opt):
    if inp != None:
        return inp
    else:
        return opt
