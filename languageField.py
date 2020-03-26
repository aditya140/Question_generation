class LanguageIndex():
    def __init__(self, lang):
        """ lang are the list of phrases from each language"""
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        
        self.create_index()
        
    def create_index(self):
        for phrase in self.lang:
            # update with individual tokens
            self.vocab.update(phrase.split(' '))
            
        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0
        self.word2idx['<pad>'] = 0
        
        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word        
    def encode(self,input):
        return [self.word2idx[s] for s in input.split(" ")]
    def decode(self,input):
        return [self.idx2word[s] for s in input]
    def vocab_size(self):
        return len(self.word2idx)