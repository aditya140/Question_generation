import wget
import json
from tqdm import tqdm
import os

class QGenDataset(object):
    def __init__(self,squad=True,USE_ENTIRE_SENTENCE=True):
        self.USE_ENTIRE_SENTENCE=USE_ENTIRE_SENTENCE
        if squad:
            if not os.path.exists("./train-v2.0.json"):
                wget.download("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")
            with open("./train-v2.0.json",'r') as f:
                self.raw_data=json.load(f)
            self.__get_CAQ__()
            
            
    def __get_CAQ__(self):
        questions=[]
        context=[]
        answers_partial=[]
        answers=[]
        title=[]
        _id=[]
        for rc in self.raw_data["data"]:
            for para in rc["paragraphs"]:
                for qas in para["qas"]:
                    if qas["answers"]!=[]:
                        questions.append(qas["question"])
                        _id.append(qas["id"])
                        context.append(para["context"])
                        title.append(rc["title"])
                        answers_partial.append( (qas["answers"][0]["text"] ,qas["answers"][0]["answer_start"]))

        assert(len(context)==len(answers_partial)==len(questions)==len(title)==len(_id))
        self.data_len=len(context)
        if self.USE_ENTIRE_SENTENCE:
            for i in (range(self.data_len)):
                answers.append(self.find_answer_sentence(answers_partial[i],context[i]))
        else:
            for i in (range(self.data_len)):
                answers.append(answers_partial[i][0])
        del answers_partial
        self.context=context
        self.questions=questions
        self.answers=answers

    def get_CAQ(self):
        return self.context,self.questions,self.answers

    def find_answer_sentence(self,answer,context,find_start_period=True):
        start_pos=answer[1]
        if find_start_period:
            while (start_pos!=0) and (context[start_pos]!="."):
                start_pos-=1
        end_pos=answer[1]+len(answer[0])
        while (end_pos<len(context)) and (context[end_pos]!="."):
            end_pos+=1
        return context[start_pos+1:end_pos]
    def __len__(self):
        return self.data_len
    def apply(self,function,all=True):
        for i in tqdm(range(self.data_len),position=0,leave=True):
            self.context[i]=function(self.context[i])
            self.answers[i]=function(self.answers[i])
            self.questions[i]=function(self.questions[i])

    def bert_format(self):
        X=[0 for i in range(self.data_len)]
        Y=[0 for i in range(self.data_len)]
        for i in range(self.data_len):
            X[i]="[CLS] " + self.context[i] +"[SEP]"+ self.answers[i] + "[SEP]"
            Y[i]=self.questions[i]
        return (X,Y)



