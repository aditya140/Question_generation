import wget
import json
from tqdm import tqdm
import os
import torchtext
import spacy
import zipfile
import unicodedata
import re
from sklearn.model_selection import train_test_split
import pandas as pd
from .languageField import LanguageIndex

from torch.utils.data import Dataset, DataLoader
import numpy as np
from CacheFunc.man import cache_it


class TrainData(Dataset):
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


class TestData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


class QGenDataset(object):
    def __init__(self):
        pass

    def __get_NMT__(self):
        original_word_pairs = [[w for w in l.split("\t")] for l in self.nmt_raw]
        self.eng = [i[0] for i in original_word_pairs]
        self.spa = [i[1] for i in original_word_pairs]

    def get_AQ(self, max_len=80, sample=True):
        raw_data = {
            "ans": [line[0] for line in self.data],
            "que": [line[1] for line in self.data],
        }
        df = pd.DataFrame(raw_data, columns=["ans", "que"])
        # remove very long sentences and sentences where translations are
        # not of roughly equal length
        df["ans_len"] = df["ans"].str.count(" ")
        df["que_len"] = df["que"].str.count(" ")
        df = df.query("ans_len <" + str(max_len) + " & que_len <" + str(max_len))
        df = df.drop_duplicates()
        if sample:
            return df["ans"].values[:2000], df["que"].values[:2000]
        return df["ans"].values, df["que"].values

    def get_NMT(self, sample=False):
        if sample:
            return self.eng[:2000], self.spa[:2000]
        return self.eng, self.spa

    def _create_dataset(self, data, normalize=True):
        load_failure = 0
        try:
            if "data" in data.keys():
                data = data["data"]
        except:
            pass
        que_ans = []
        for topic in data:
            for para in topic["paragraphs"]:
                for qa in para["qas"]:
                    try:
                        res = []
                        if normalize:
                            res.append(
                                self._normalize(
                                    self._get_sentence(
                                        para["context"],
                                        qa["answers"][0]["answer_start"],
                                        qa["answers"][0]["text"],
                                    )
                                )
                            )
                            res.append(self._normalize(qa["question"]))
                        else:
                            res.append(
                                self._get_sentence(
                                    para["context"],
                                    qa["answers"][0]["answer_start"],
                                    qa["answers"][0]["text"],
                                )
                            )
                            res.append(qa["question"])
                        que_ans.append(res)
                    except:
                        load_failure += 1
        print("Load Failure : ", load_failure)
        return que_ans

    @staticmethod
    def _get_sentence(context, position, text):
        if "." in text[:-1]:
            return_2 = True
        else:
            return_2 = False
        context = context.split(".")
        count = 0
        for sent in range(len(context)):
            if count + len(context[sent]) > position:
                if return_2:
                    return ".".join(context[sent : sent + 2])
                else:
                    return context[sent]
            else:
                count += len(context[sent]) + 1
        return False

    def _get_dataset(self, normalize=True):
        data = self._create_dataset(self.raw_data, normalize=normalize)
        return data

    def __len__(self):
        return self.data_len

    def apply(self, function, all=True):
        for i in tqdm(range(self.data_len), position=0, leave=True):
            self.context[i] = function(self.context[i])
            self.answers[i] = function(self.answers[i])
            self.questions[i] = function(self.questions[i])

    def bert_format(self):
        X = [0 for i in range(self.data_len)]
        Y = [0 for i in range(self.data_len)]
        for i in range(self.data_len):
            X[i] = "[CLS] " + self.context[i] + "[SEP]" + self.answers[i] + "[SEP]"
            Y[i] = self.questions[i]
        return (X, Y)

    @staticmethod
    def unicodeToAscii(s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    def _normalize(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def initialize(self,):
        if self.squad:
            if not os.path.exists("./train-v2.0.json"):
                wget.download(
                    "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
                )
            with open("./train-v2.0.json", "r") as f:
                self.raw_data = json.load(f)
            self.data = self._get_dataset()
        if not self.squad:
            if not os.path.exists("./spa-eng.zip"):
                wget.download("http://download.tensorflow.org/data/spa-eng.zip")
            if not os.path.exists("./spa/spa-eng/spa.txt"):
                with zipfile.ZipFile("./spa-eng.zip", "r") as zip_ref:
                    zip_ref.extractall("./spa/")
            with open("./spa/spa-eng/spa.txt", "r") as f:
                self.nmt_raw = f.read().strip().split("\n")
            self.__get_NMT__()

    @cache_it
    def getData(
        self,
        input_vocab,
        output_vocab,
        max_len,
        tokenizer,
        sample=False,
        batch_size=64,
        val_split=0.1,
        test_split=0.1,
        squad=True,
        USE_ENTIRE_SENTENCE=True,
    ):
        """[summary]

        Arguments:
            input_vocab {[type]} -- [description]
            output_vocab {[type]} -- [description]
            max_len {[type]} -- [description]
            tokenizer {[type]} -- [description]

        Keyword Arguments:
            sample {bool} -- [description] (default: {False})
            batch_size {int} -- [description] (default: {64})
            val_split {float} -- [description] (default: {0.1})
            test_split {float} -- [description] (default: {0.1})

        Returns:
            train_dataset[DataLoader] -- [description]
            val_dataset[DataLoader] -- [description]
            test_dataset[DataLoader] -- [description]
            inpLang[LanguageIndex] -- [description]
            optLang[LanguageIndex] -- [description]
            [type] -- [description]
        """
        self.USE_ENTIRE_SENTENCE = USE_ENTIRE_SENTENCE
        self.squad = squad

        self.initialize()
        args = [
            input_vocab,
            output_vocab,
            max_len,
            tokenizer,
            sample,
            batch_size,
            val_split,
            test_split,
        ]

        if self.squad:
            input_, output_ = self.get_AQ(max_len=max_len, sample=sample)
        else:
            input_, output_ = self.get_NMT(sample=sample)
        print(f"Loaded: {len(input_)} samples")
        (
            train_set_input,
            test_set_input,
            train_set_output,
            test_set_output,
        ) = train_test_split(input_, output_, test_size=test_split)
        input_train, input_test, output_train, output_test = train_test_split(
            train_set_input, train_set_output, test_size=val_split
        )
        inpLang = LanguageIndex(
            input_train, vocab_size=input_vocab, max_len=max_len, tokenizer=tokenizer
        )
        optLang = LanguageIndex(
            output_train, vocab_size=output_vocab, max_len=max_len, tokenizer=tokenizer
        )
        input_train_tokens = inpLang.encode_batch(input_train)
        input_test_tokens = inpLang.encode_batch(input_test)
        ouptut_train_tokens = optLang.encode_batch(output_train)
        output_test_tokens = optLang.encode_batch(output_test)

        test_dataset = TestData(test_set_input, test_set_output)
        train_dataset = TrainData(input_train_tokens, ouptut_train_tokens)
        val_dataset = TrainData(input_test_tokens, output_test_tokens)
        return train_dataset, val_dataset, test_dataset, inpLang, optLang
