nvidia-smi
python3 test_cuda.py
python3 src/trainProcs/transformer/transformer.py --gpu --to_artifact --epochs 1 | tee /artifacts/test_log.log