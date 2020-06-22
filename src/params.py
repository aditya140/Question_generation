import argparse


__SEQ2SEQ__ = {
    "save_path": "./src/saved_models/seq2seq",
    "lr": 1e-3,
    "input_vocab": 45000,
    "output_vocab": 45000,
    "embedding_dim": 300,
    "pretrained": True,
    "rnn_units": 2,
    "hidden_size": 600,
    "batch_size": 64,
    "squad": True,
    "tokenizer": "spacy",
    "max_len": 40,
    "sample": False,
    "dropout": 0.3,
    "val_split": 0.1,
    "test_split": 0.1,
    "enc_emb_dim": 300,
    "dec_emb_dim": 300,
    "enc_dropout": 0.3,
    "dec_dropout": 0.3,
    "epochs": 8,
    "optim":"SGD",
    "scheduler":True,
    "trial_path": "./src/experiment/seq2seq/",
    "to_artifact": False,
    "prune": True,
}
SEQ2SEQ_PARAMS = argparse.Namespace(**__SEQ2SEQ__)

__TRANSFORMER__ = {
    "save_path": "./src/saved_models/transformer",
    "lr": 1e-3,
    "usetpu": False,
    "squad": True,
    "sample": False,
    "epochs": 13,
    "input_vocab": 80000,
    "output_vocab": 80000,
    "max_len": 100,
    "batch_size": 64,
    "tokenizer": "BERT",
    "model_name": "transformer",
    "hidden_dim": 256,
    "enc_layers": 3,
    "dec_layers": 3,
    "enc_heads": 8,
    "dec_heads": 8,
    "enc_pf_dim": 600,
    "dec_pf_dim": 600,
    "enc_dropout": 0.1,
    "dec_dropout": 0.1,
    "val_split": 0.1,
    "test_split": 0.1,
    "to_artifact": False,
    "auto_lr_find": False,
}
TRANSFORMER_PARAMS = argparse.Namespace(**__TRANSFORMER__)
