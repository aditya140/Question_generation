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
"usetpu":False,
"SQUAD":True,
"SAMPLE":False,
"EPOCHS":8,
"INPUT_VOCAB":80000,
"OUTPUT_VOCAB":40000,
"MAX_LEN":100,
"BATCH_SIZE":128,
"EMB_DIM":300,
"tokenizer":"spacy",
"lr":1e-3,
"model_name":"transformer",
"HID_DIM" : 256,
"ENC_LAYERS" : 3,
"DEC_LAYERS" : 3,
"ENC_HEADS" : 8,
"DEC_HEADS" : 8,
"ENC_PF_DIM" : 600,
"DEC_PF_DIM" : 600,
"ENC_DROPOUT" : 0.1,
"DEC_DROPOUT" : 0.1
}
TRANSFORMER_PARAMS = argparse.Namespace(**__TRANSFORMER__)
