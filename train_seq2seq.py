from data_loader import QGenDataset
from seq2seq import *
from languageField import LanguageIndex
from pprint import pprint
from utils import *
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import time
import math
from torch.utils.data import Dataset, DataLoader
from seq2seq_params import *
from hyperdash import Experiment




def train(model,iterator,optimizer,loss_func,device,verbose=False):
    model.train()
    epoch_loss=0
    iters=len(iterator)
    print(f"Total iterations {iters}")
    for i,(x,y,x_len) in enumerate(iterator):
        optimizer.zero_grad()
        opt=model(x,y)
        output_dim = opt.shape[-1]
        opt = opt.view(-1, output_dim)
        loss=loss_func(opt,y.view(-1).to(device))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss+=loss.cpu().item()
        if HYPERDASH:
            exp.metric("train_loss",loss.item())
            exp.metric("train_iter",i)
        if verbose:
            print(f"iter {i}/{iters} loss : ",loss.item())
    avg_epoch_loss=epoch_loss/len(iterator)
    return avg_epoch_loss

def evaluate(model,iterator,loss_func,device):
    epoch_loss=0
    model.eval()
    for i,(x,y,x_len) in enumerate(iterator):
        opt=model(x,y,teacher_forcing=False)
        output_dim = opt.shape[-1]
        opt = opt.view(-1, output_dim)
        loss=loss_func(opt,y.view(-1).to(device))
        epoch_loss+=loss.cpu().item()
        if HYPERDASH:
            exp.metric("val_loss",loss.item())
        #test_infer
    print(" ".join(model.infer(random.choice(answers),ansLang,quesLang)))
    avg_epoch_loss=epoch_loss/len(iterator)
    return avg_epoch_loss


if __name__=="__main__":
    qg=QGenDataset()
    _,questions,answers=qg.get_CAQ(sample=SAMPLE)
    ques_train,ques_test,ans_train,ans_test=train_test_split(questions,answers,test_size=0.2)
    quesLang=LanguageIndex(ques_train,vocab_size=INPUT_VOCAB,max_len=MAX_LEN)
    ansLang=LanguageIndex(ans_train,vocab_size=OUTPUT_VOCAB,max_len=MAX_LEN)

    ques_train_tokens=quesLang.encode_batch(ques_train)
    ques_test_tokens=quesLang.encode_batch(ques_test)
    ans_train_tokens=ansLang.encode_batch(ans_train)
    ans_test_tokens=ansLang.encode_batch(ans_test)

    train_dataset = MyData(ans_train_tokens,ques_train_tokens)
    test_dataset = MyData(ans_test_tokens, ques_test_tokens)

    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                        drop_last=True,
                        shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, 
                        drop_last=True,
                        shuffle=True)


    # LOG on HYPERDASH
    if HYPERDASH:
        exp = Experiment("seq2seq")
        exp.param('INPUT_VOCAB',INPUT_VOCAB)
        exp.param('OUTPUT_VOCAB',OUTPUT_VOCAB)
        exp.param('MAX_LEN',MAX_LEN)
        exp.param('BATCH_SIZE',BATCH_SIZE)
        exp.param('EMBEDDING_DIM',EMBEDDING_DIM)
        exp.param('UNITS',UNITS)
        exp.param('HIDDEN_SIZE',HIDDEN_SIZE)
        exp.param('EPOCHS',EPOCHS)
        exp.param('DEVICE',DEVICE)
        exp.param('LR',LR)


    torch.cuda.empty_cache()


    seq2seq=Seq2seq(input_vocab=INPUT_VOCAB,output_vocab=OUTPUT_VOCAB,embedding_dim=EMBEDDING_DIM,rnn_units=UNITS,hidden_size=HIDDEN_SIZE,batch_size=BATCH_SIZE,teacher_forcing=TEACHER_FORCING,device=device,bidirectional=BIDIRECTIONAL).to(device)
    seq2seq.init_weights()
    loss_func=nn.CrossEntropyLoss()
    optimizer=optim.Adam(seq2seq.parameters(),lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' , patience=2)
    best_model = float('inf')
    for ep in range(EPOCHS):
        print(f"EPOCH {ep} :")
        start=time.time()
        if HYPERDASH:
            exp.metric("EPOCH",ep)
        train_loss=train(seq2seq,train_dataloader,optimizer,loss_func,device,verbose=True)
        val_loss=evaluate(seq2seq,test_dataloader,loss_func,device)
        time_taken=time.time()-start
        scheduler.step(val_loss)
        print(f"\tTraining Loss = {train_loss}")
        print(f"\tValidation Loss = {val_loss}")
        print(f"\tTime per Epoch = {int(time_taken/60)} mins {int(time_taken) - int(time_taken/60)*60}\n")
        if val_loss < best_model:
            best_model = val_loss
            torch.save(seq2seq.state_dict(), './seq2seq.pt')

    if HYPERDASH:
        exp.end()