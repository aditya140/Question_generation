import torch

INPUT_VOCAB=80000
OUTPUT_VOCAB=40000
MAX_LEN=80
BATCH_SIZE=64
EMBEDDING_DIM=300
UNITS=1
HIDDEN_SIZE=200
EPOCHS=1
DEVICE="GPU"
HYPERDASH=False
LR=0.00008
TEACHER_FORCING=0.8
BIDIRECTIONAL=True
SAMPLE=True

if DEVICE=="CPU":
    device=torch.device('cpu')
else:
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))