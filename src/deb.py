from dataPrep.data_loader import QGenDataset

a=QGenDataset()
data=a.getData(input_vocab=10000, output_vocab=10000, max_len=100, tokenizer="spacy", sample=False, batch_size=64, val_split=0.1, test_split=0.1)
print(len(data))