from models.seq2seq import Seq2seq
import pytorch_lightning as pl


class Seq2seq_pl(pl.LightningModule):
    def __init__(self,):
        super(Seq2seq,self).__init__()
        self.model=Seq2seq()