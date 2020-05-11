import torch
DEVICE= "GPU"
testNMT=False
SAMPLE=True
MAX_LEN=60
BATCH_SIZE=128
USE_PRETRAINED=False
TOKENIZER="spacy"
INPUT_VOCAB=80000
OUTPUT_VOCAB=40000
EMBEDDING_DIM=300
UNITS=1
HIDDEN_SIZE=200
EPOCHS=1
HYPERDASH=False
LR=0.00008
TEACHER_FORCING=0.8
BIDIRECTIONAL=True
SAMPLE=True
GLOVE_PATH="/content/drive/My Drive/glove.p"
TEST_BATCH_SIZE=20
LAYERS=1

if TOKENIZER=="BERT":
    trainNMT=False
    USE_PRETRAINED=False
if testNMT:
    USE_PRETRAINED=False

if DEVICE=="CPU":
    device=torch.device('cpu')
else:
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))