nvidia-smi
python3 test_cuda.py
python3 src/trainProcs/transformer/transformer.py --gpu --to_artifact --epochs 1 --sample| tee /artifacts/test_log.log
python3 src/trainProcs/seq2seq/seq2seq.py --gpu --to_artifact --epochs 1 --sample | tee /artifacts/test_log.log
