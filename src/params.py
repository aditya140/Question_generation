import argparse


__SEQ2SEQ__={
    "save_path":"./src/saved_model/seq2seq/seq2seq.pt",
    "lr":1e-3,
    "input_vocab":30000,
    "output_vocab":60000,
    "embedding_dim":300,
    "rnn_units":1,
    "hidden_size":128,
    "batch_size":64,
    "squad":True,
    "tokenizer":"spacy",
    "max_len":80,
    "sample":False,
    "dropout":0.3,
    "val_split":0.1,
    "test_split":0.1,
    "enc_emb_dim":300,
    "dec_emb_dim":300,
    "enc_dropout":0.3,
    "dec_dropout":0.3,
    "epochs":1,
    "trial_path":"./src/experiment/seq2seq/",
    "prune":True
}
SEQ2SEQ_PARAMS=argparse.Namespace(**__SEQ2SEQ__)

