import argparse


__SEQ2SEQ__ = {
    "save_path": "./src/saved_models/seq2seq",
    "lr": 1e-3,
    "input_vocab": 10000,
    "output_vocab": 10000,
    "embedding_dim": 300,
    "rnn_units": 2,
    "hidden_size": 128,
    "batch_size": 64,
    "squad": False,
    "tokenizer": "spacy",
    "max_len": 10,
    "sample": False,
    "dropout": 0.3,
    "val_split": 0.1,
    "test_split": 0.1,
    "enc_emb_dim": 256,
    "dec_emb_dim": 256,
    "enc_dropout": 0.3,
    "dec_dropout": 0.3,
    "epochs": 8,
    "trial_path": "./src/experiment/seq2seq/",
    "prune": True,
}
SEQ2SEQ_PARAMS = argparse.Namespace(**__SEQ2SEQ__)

__TRANSFORMER__ =  {
    "save_path": "./src/saved_models/transformer",
    "lr": 5e-4,
    "usetpu":False,
    "squad": True,
    "sample": False,
    "epochs":13,
    "input_vocab":80000,
    "output_vocab":40000,
    "max_len":100,
    "batch_size": 64,
    "tokenizer":"spacy",
    "model_name":"transformer",
    "hidden_dim" : 512,
    "enc_layers" : 5,
    "dec_layers" : 5,
    "enc_heads" : 16,
    "dec_heads" : 16,
    "enc_pf_dim" : 600,
    "dec_pf_dim" : 600,
    "enc_dropout" : 0.1,
    "dec_dropout" : 0.1,
    "val_split": 0.1,
    "test_split": 0.1,
}
TRANSFORMER_PARAMS = argparse.Namespace(**__TRANSFORMER__)
