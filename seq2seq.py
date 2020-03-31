from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import torch
class Encoder(nn.Module):
    def __init__(self,vocab_size,embedding_dim,encoder_units,hidden_size,batch_size,bidirectional):
        super(Encoder,self).__init__()
        self.batch_size=batch_size 
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.bidirectional=bidirectional
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim,padding_idx=0) #[emb]
        self.rnn=nn.GRU(input_size=embedding_dim,hidden_size=hidden_size,num_layers=encoder_units,dropout=0.2,bidirectional=bidirectional)

    def forward(self,input,device):
        x=input.transpose(0,1) # x = [max_len , batch_size ]
        x=self.embedding(x)  # x = [max_len , batch_size , emb_dim]
        batch_size=x.shape[1]
        hidden=torch.zeros(2 if self.bidirectional else 1,batch_size,self.hidden_size).to(device) # hidden = [1 , batch_size , hidden_size]
        output,hidden=self.rnn(x,hidden) # output = [max_len , batch_size , emb_dim]  hidden = [1 , batch_size , hidden_size]
        return output,hidden



class Decoder(nn.Module):
    def __init__(self,vocab_size,embedding_dim,decoder_units,hidden_size,batch_size,bidirectional):
        super(Decoder,self).__init__()
        self.batch_size=batch_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.bidirectional=bidirectional
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim,padding_idx=0)
        self.rnn=nn.GRU(input_size=embedding_dim,hidden_size=hidden_size,num_layers=decoder_units,dropout=0.2,bidirectional=BIDIRECTIONAL)
        self.fc=nn.Linear(hidden_size*(2 if self.bidirectional else 1),vocab_size)

    def forward(self,input,hidden,device): 
        x=input.view(1,-1) # x = [1 , batch_size ]
        x=self.embedding(x) # x = [1 , batch_size , emb_dim]
        output, hidden = self.rnn(x, hidden)
        prediction = self.fc(output)
        return prediction, output ,hidden




class Seq2seq(nn.Module):
    def __init__(self,input_vocab,output_vocab,embedding_dim,rnn_units,hidden_size,batch_size,teacher_forcing,device,bidirectional):
        super(Seq2seq,self).__init__()
        self.input_vocab=input_vocab
        self.output_vocab=output_vocab
        self.batch_size=batch_size
        self.teacher_forcing=teacher_forcing
        self.encoder=Encoder(input_vocab,embedding_dim=embedding_dim,encoder_units=rnn_units,hidden_size=hidden_size,batch_size=batch_size,bidirectional=bidirectional)
        self.decoder=Decoder(output_vocab,embedding_dim=embedding_dim,decoder_units=rnn_units,hidden_size=hidden_size,batch_size=batch_size,bidirectional=bidirectional)
        self.device=device

    def forward(self,input,target,teacher_forcing=True):
        input=input.to(self.device)
        target=target.to(self.device)
        # target_len=sum(torch.sum(target,dim=0)==0).item()
        target_len=target.shape[-1]
        # print(target_len)
        final_opt=torch.zeros(target_len, self.batch_size, self.output_vocab).to(self.device)
        enc_output,enc_hidden=self.encoder(input,self.device)
        decoder_input=target[:,0]
        for t in range(1,target_len):
            pred,dec_opt,dec_hidden=self.decoder(decoder_input,enc_hidden,self.device)
            enc_hidden=dec_hidden
            top1=pred.argmax(dim=2)
            final_opt[t-1]=pred
            decoder_input=target[:,t]
            if teacher_forcing:
                teacher_force=random.random()<self.teacher_forcing
                if teacher_force:
                    decoder_input=top1                
        return final_opt

    def infer(self,input,target_lang,input_lang,max_size=40):
        input=input_lang.encode(input)
        input=torch.tensor(input).to(self.device).view(1,-1)
        output=[]
        enc_output,hidden=self.encoder(input,self.device)
        decoder_input=torch.tensor([target_lang.word2idx[target_lang.special["init_token"]]]).to(self.device)
        out_word=target_lang.special["init_token"]
        while out_word!=target_lang.special["eos_token"] and len(output)<=max_size:
            pred,dec_opt,dec_hidden=self.decoder(decoder_input,hidden,self.device)
            hidden=dec_hidden
            topk=pred.squeeze(1).topk(5)
            topk_index=topk.indices.squeeze(0).tolist()
            ind=0
            while ((topk_index[ind] in [target_lang.word2idx[target_lang.special["pad_token"]],target_lang.word2idx[target_lang.special["init_token"]]]) or (len(output)<3 and topk_index[ind] in [target_lang.word2idx[target_lang.special["eos_token"]],target_lang.word2idx[target_lang.special["init_token"]]])):
                ind+=1
            top1_index=topk_index[ind]
            try:
                out_word=target_lang.idx2word[top1_index]
            except:
                out_word=target_lang.special["unk_token"]
            output.append(out_word)
            decoder_inp=pred[0,0,top1_index]
        return output


    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)