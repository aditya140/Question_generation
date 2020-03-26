from data_loader import QGenDataset,unicode_to_ascii,preprocess_sentence
from vanilla_seq2seq import Seq2Seq,Encoder,Decoder
from languageField import LanguageIndex
from pprint import pprint
from utils import *
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import time
import math
from torch.utils.data import Dataset, DataLoader


# conver the data to tensors and pass to the Dataloader 
# to create an batch iterator

class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        # TODO: convert this into torch code is possible
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x,y,x_len
    
    def __len__(self):
        return len(self.data)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for (i, (inp, targ, inp_len)) in enumerate(iterator):
        print(inp,)
        src = inp
        trg = targ
        optimizer.zero_grad()
        output = model(src, trg)
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for (i, (inp, targ, inp_len)) in enumerate(iterator):
            src = inp
            trg = targ
            output = model(src, trg, 0) #turn off teacher forcing
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg) 
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_data():
    dataset=QGenDataset(squad=True)
    dataset.apply(preprocess_sentence)
    _,questions,answers=dataset.get_CAQ()
    return questions,answers

def prepare_langugage(text):
    lang = LanguageIndex(text)
    # Vectorize the input and target languages
    text_tensor = [lang.encode(sent)  for sent in text]
    return lang,text_tensor

def split_data(input_tensor,target_tensor,test_size=0.2):
    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=test_size)
    # Show length
    print("Length of Data",len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
    return input_tensor_train,target_tensor_train,input_tensor_val,target_tensor_val

if __name__=="__main__":
    dataset=QGenDataset(squad=True)
    dataset.apply(preprocess_sentence)
    _,questions,answers=dataset.get_CAQ()
    ans_lang,ans_tensor=prepare_langugage(answers)
    ques_lang,ques_tensor=prepare_langugage(questions)
    max_length_inp, max_length_tar = max_length(ans_tensor), max_length(ques_tensor)
    print(f"Max input length: {max_length_inp}")
    print(f"Max output length: {max_length_tar}")
    input_tensor = [pad_sequences(x, max_length_inp) for x in ans_tensor]
    target_tensor = [pad_sequences(x, max_length_tar) for x in ques_tensor]
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = split_data(input_tensor, target_tensor, test_size=0.2)
    
    """
    HYPERPARAMETERS
    """
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dim = 256
    units = 1024
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    N_LAYERS=2

    """
    
    """
    vocab_inp_size = ans_lang.vocab_size()
    vocab_tar_size = ques_lang.vocab_size()
    print(f"Input vocab size: {vocab_inp_size}")
    print(f"Output vocab size: {vocab_tar_size}")



    train_dataset = MyData(input_tensor_train, target_tensor_train)
    val_dataset = MyData(input_tensor_val, target_tensor_val)

    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last=True,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last=True,shuffle=True)

    it = iter(train_dataloader)
    x, y, x_len = next(it)
    print(f"Data Loader first iter = {x_len}")

    device = get_torch_device()


    enc = Encoder(vocab_inp_size, embedding_dim, units, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(vocab_tar_size, embedding_dim, units, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    model.init_weights()
    print(f'The model has {model.count_parameters()} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index = ques_lang.word2idx["<pad>"])
    N_EPOCHS=1
    CLIP = 1
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_dataloader, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'vanilla-seq2seq-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')








    

        

