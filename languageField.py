import spacy
from collections import Counter 
import sys
import numpy as np
class LanguageIndex():

    def __init__(self, lang, pad="<PAD>",init_token="<SOS>",eos_token="<EOS>",unk_token="<UNK>",max_len=None,vocab_size=None,lower_case=True):
        """ lang are the list of phrases from each language"""
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.special={}
        self.max_len=max_len
        self.vocab_size=vocab_size if vocab_size!=None else sys.maxsize
        self.lower=lower_case

        # add a padding token with index 0
        self.word2idx[pad] = 0
        self.special["pad_token"]=pad

        self.word2idx[init_token] = 1
        self.special["init_token"]=init_token

        self.word2idx[eos_token] = 2
        self.special["eos_token"]=eos_token

        self.word2idx[unk_token] = 3
        self.special["unk_token"]=unk_token

        self.vocab = set()
        self.counter=Counter()
        self.spacy=None
        self.create_index()

    @staticmethod
    def unicode_to_ascii(s):
        """
        Normalizes latin chars with accent to their canonical decomposition
        """
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
        
    @staticmethod
    def preprocess_sentence(w):
        w = unicode_to_ascii(w.lower().strip())
        
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ." 
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        
        w = w.rstrip().strip()
        
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        return w
        
    def tokenize(self,phrase,tokenizer="spacy"):
        if tokenizer=="spacy":
            if not self.spacy:
                self.spacy = spacy.load('en')
            return [tok.text for tok in self.spacy.tokenizer(phrase)]
        else:
            return self.preprocess(phrase)

    def create_index(self):
        for phrase in self.lang:
            # update with individual tokens
            tokens=self.tokenize(phrase.lower() if self.lower else phrase)
            self.vocab.update(tokens)
            self.counter.update(tokens)
            
        # sort the vocab
        self.vocab = sorted(self.vocab)
        start_index = max(self.word2idx.values())+1
        
        # word to index mapping
        for index, word in enumerate(self.counter.most_common(self.vocab_size)):
            self.word2idx[word[0]] = index + start_index 
        
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

    def encode_batch(self,batch,special_tokens=True):
        return np.array([self.encode(obj,special_tokens=special_tokens) for obj in batch],dtype=np.int64)
    def decode_batch(self,batch):
        return [self.decode(obj) for obj in batch]

    def encode(self,input,special_tokens=True):
        pad_len=self.max_len
        input=input.lower() if self.lower else input
        tokens=[tok.text for tok in self.spacy.tokenizer(input)]
        if pad_len!=None:
            if len(tokens)>pad_len-(2 if special_tokens else 0):
                if special_tokens:
                    return [1]+[self.word2idx[s] if s in self.word2idx.keys() else 3 for s in tokens][:pad_len-2]+[2]
                else:
                    return [self.word2idx[s] if s in self.word2idx.keys() else 3 for s in tokens][:pad_len]
            else:
                return ([1] if special_tokens else []) + [self.word2idx[s] if s in self.word2idx.keys() else 3 for s in tokens] +([2] if special_tokens else []) +[0 for i in range(pad_len-(2 if special_tokens else 0)-len(tokens))]
        return ([1] if special_tokens else []) + [self.word2idx[s] if s in self.word2idx.keys() else 3 for s in tokens] +([2] if special_tokens else []) 
    def decode(self,input):
        return [self.idx2word[s] for s in input]
    def vocab_size(self):
        return len(self.word2idx)