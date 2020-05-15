import argparse


__SEQ2SEQ__={
    "save_path":"./src/saved_models/seq2seq",
    "lr":1e-3,
    "input_vocab":60000,
    "output_vocab":30000,
    "embedding_dim":300,
    "rnn_units":1,
    "hidden_size":128,
    "batch_size":64,
    "squad":False,
    "tokenizer":"spacy",
    "max_len":20,
    "sample":False,
    "dropout":0.3,
    "val_split":0.1,
    "test_split":0.1,
    "enc_emb_dim":256,
    "dec_emb_dim":256,
    "enc_dropout":0.3,
    "dec_dropout":0.3,
    "epochs":2,
    "trial_path":"./src/experiment/seq2seq/",
    "prune":True
}
SEQ2SEQ_PARAMS=argparse.Namespace(**__SEQ2SEQ__)

