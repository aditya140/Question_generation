from dataPrep.dataset import QGenDataset
from torch.utils.data import DataLoader

class SimpleDataloader:
    """[summary]
    Simple Dataloader for Question Generation
    """    
    def __init__(
        self,
        input_vocab,
        output_vocab,
        max_len,
        tokenizer,
        sample,
        batch_size,
        val_split,
        test_split,
        squad,
        **kwargs,
    ):
        """[summary]

        Arguments:
            input_vocab {[type]} -- [description]
            output_vocab {[type]} -- [description]
            max_len {[type]} -- [description]
            tokenizer {[type]} -- [description]
            sample {[type]} -- [description]
            batch_size {[type]} -- [description]
            val_split {[type]} -- [description]
            test_split {[type]} -- [description]
            squad {[type]} -- [description]
        """        
        self.batch_size=batch_size
        self.qg = QGenDataset()
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.inpLang,
            self.optLang,
        ) = self.qg.getData(
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            max_len=max_len,
            tokenizer=tokenizer,
            sample=sample,
            batch_size=batch_size,
            squad=squad,
        )

    def get_train_dataloader(self,):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=10)
    def get_test_dataloader(self,):
        return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=10)
    def get_val_dataloader(self,):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=10)

